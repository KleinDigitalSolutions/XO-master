-- Klein Digital Solutions - AI Audio Services
-- Secure Credit System Database Schema
-- Supabase PostgreSQL with Row Level Security (RLS)

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================
-- TABLES
-- =============================================

-- Users table for credit management
CREATE TABLE IF NOT EXISTS credit_users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    credits INTEGER DEFAULT 0 CHECK (credits >= 0),
    stripe_customer_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Credit purchases (from Stripe payments)
CREATE TABLE IF NOT EXISTS credit_purchases (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id TEXT REFERENCES credit_users(id) ON DELETE CASCADE,
    credits INTEGER NOT NULL CHECK (credits > 0),
    package_type TEXT NOT NULL,
    stripe_session_id TEXT UNIQUE,
    payment_amount DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Credit usage tracking
CREATE TABLE IF NOT EXISTS credit_usage (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id TEXT REFERENCES credit_users(id) ON DELETE CASCADE,
    credits INTEGER NOT NULL CHECK (credits > 0),
    service TEXT NOT NULL,
    job_id TEXT,
    modal_endpoint TEXT,
    processing_time INTEGER, -- in seconds
    file_size INTEGER, -- in bytes
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================
-- INDEXES for Performance
-- =============================================

CREATE INDEX IF NOT EXISTS idx_credit_users_email ON credit_users(email);
CREATE INDEX IF NOT EXISTS idx_credit_purchases_user_id ON credit_purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_credit_purchases_session_id ON credit_purchases(stripe_session_id);
CREATE INDEX IF NOT EXISTS idx_credit_usage_user_id ON credit_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_credit_usage_service ON credit_usage(service);
CREATE INDEX IF NOT EXISTS idx_credit_usage_created_at ON credit_usage(created_at DESC);

-- =============================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================

-- Enable RLS on all tables
ALTER TABLE credit_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_purchases ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_usage ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own profile" ON credit_users
    FOR SELECT USING (auth.uid()::text = id OR auth.role() = 'service_role');

CREATE POLICY "Users can update own profile" ON credit_users
    FOR UPDATE USING (auth.uid()::text = id OR auth.role() = 'service_role');

-- Only service role can insert/delete users (via API)
CREATE POLICY "Service role can manage users" ON credit_users
    FOR ALL USING (auth.role() = 'service_role');

-- Users can view their own purchases
CREATE POLICY "Users can view own purchases" ON credit_purchases
    FOR SELECT USING (auth.uid()::text = user_id OR auth.role() = 'service_role');

-- Only service role can insert purchases (via webhook)
CREATE POLICY "Service role can manage purchases" ON credit_purchases
    FOR ALL USING (auth.role() = 'service_role');

-- Users can view their own usage
CREATE POLICY "Users can view own usage" ON credit_usage
    FOR SELECT USING (auth.uid()::text = user_id OR auth.role() = 'service_role');

-- Only service role can manage usage
CREATE POLICY "Service role can manage usage" ON credit_usage
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================
-- TRIGGERS
-- =============================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to credit_users
CREATE TRIGGER update_credit_users_updated_at
    BEFORE UPDATE ON credit_users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================
-- STORED PROCEDURES/FUNCTIONS
-- =============================================

-- Function to add credits (atomic transaction)
CREATE OR REPLACE FUNCTION add_user_credits(
    p_user_id TEXT,
    p_credits INTEGER,
    p_package_type TEXT,
    p_session_id TEXT,
    p_payment_amount DECIMAL
)
RETURNS TABLE(new_balance INTEGER, transaction_id UUID) AS $$
DECLARE
    v_transaction_id UUID;
    v_new_balance INTEGER;
BEGIN
    -- Start transaction
    BEGIN
        -- Insert purchase record
        INSERT INTO credit_purchases (user_id, credits, package_type, stripe_session_id, payment_amount)
        VALUES (p_user_id, p_credits, p_package_type, p_session_id, p_payment_amount)
        RETURNING id INTO v_transaction_id;
        
        -- Update user credits
        UPDATE credit_users 
        SET credits = credits + p_credits,
            updated_at = NOW()
        WHERE id = p_user_id
        RETURNING credits INTO v_new_balance;
        
        -- Return results
        RETURN QUERY SELECT v_new_balance, v_transaction_id;
        
    EXCEPTION WHEN OTHERS THEN
        -- Rollback on error
        RAISE EXCEPTION 'Failed to add credits: %', SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to deduct credits (atomic transaction)
CREATE OR REPLACE FUNCTION deduct_user_credits(
    p_user_id TEXT,
    p_credits INTEGER,
    p_service TEXT,
    p_job_id TEXT DEFAULT NULL
)
RETURNS TABLE(new_balance INTEGER, usage_id UUID) AS $$
DECLARE
    v_usage_id UUID;
    v_current_balance INTEGER;
    v_new_balance INTEGER;
BEGIN
    -- Start transaction
    BEGIN
        -- Check current balance
        SELECT credits INTO v_current_balance 
        FROM credit_users 
        WHERE id = p_user_id;
        
        -- Check if user has enough credits
        IF v_current_balance < p_credits THEN
            RAISE EXCEPTION 'Insufficient credits. Current: %, Required: %', v_current_balance, p_credits;
        END IF;
        
        -- Insert usage record
        INSERT INTO credit_usage (user_id, credits, service, job_id)
        VALUES (p_user_id, p_credits, p_service, p_job_id)
        RETURNING id INTO v_usage_id;
        
        -- Update user credits
        UPDATE credit_users 
        SET credits = credits - p_credits,
            updated_at = NOW()
        WHERE id = p_user_id
        RETURNING credits INTO v_new_balance;
        
        -- Return results
        RETURN QUERY SELECT v_new_balance, v_usage_id;
        
    EXCEPTION WHEN OTHERS THEN
        -- Rollback on error
        RAISE EXCEPTION 'Failed to deduct credits: %', SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user credit summary
CREATE OR REPLACE FUNCTION get_user_credit_summary(p_user_id TEXT)
RETURNS TABLE(
    total_credits INTEGER,
    total_purchased INTEGER,
    total_used INTEGER,
    last_purchase_date TIMESTAMPTZ,
    last_usage_date TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        u.credits,
        COALESCE(SUM(p.credits), 0)::INTEGER as total_purchased,
        COALESCE(SUM(usage.credits), 0)::INTEGER as total_used,
        MAX(p.created_at) as last_purchase,
        MAX(usage.created_at) as last_usage
    FROM credit_users u
    LEFT JOIN credit_purchases p ON u.id = p.user_id
    LEFT JOIN credit_usage usage ON u.id = usage.user_id
    WHERE u.id = p_user_id
    GROUP BY u.id, u.credits;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================
-- VIEWS for Analytics
-- =============================================

-- Daily credit usage analytics
CREATE OR REPLACE VIEW daily_credit_analytics AS
SELECT 
    DATE(created_at) as date,
    service,
    COUNT(*) as usage_count,
    SUM(credits) as total_credits_used,
    AVG(credits) as avg_credits_per_use,
    COUNT(DISTINCT user_id) as unique_users
FROM credit_usage
GROUP BY DATE(created_at), service
ORDER BY date DESC, service;

-- Monthly revenue analytics
CREATE OR REPLACE VIEW monthly_revenue_analytics AS
SELECT 
    DATE_TRUNC('month', created_at) as month,
    package_type,
    COUNT(*) as purchase_count,
    SUM(payment_amount) as total_revenue,
    SUM(credits) as total_credits_sold,
    AVG(payment_amount) as avg_purchase_value
FROM credit_purchases
GROUP BY DATE_TRUNC('month', created_at), package_type
ORDER BY month DESC, package_type;

-- User activity summary
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    u.id,
    u.email,
    u.credits as current_balance,
    COALESCE(p.total_purchased, 0) as lifetime_credits_purchased,
    COALESCE(usage.total_used, 0) as lifetime_credits_used,
    COALESCE(p.total_spent, 0) as lifetime_spent,
    p.last_purchase,
    usage.last_usage,
    u.created_at as user_since
FROM credit_users u
LEFT JOIN (
    SELECT 
        user_id,
        SUM(credits) as total_purchased,
        SUM(payment_amount) as total_spent,
        MAX(created_at) as last_purchase
    FROM credit_purchases 
    GROUP BY user_id
) p ON u.id = p.user_id
LEFT JOIN (
    SELECT 
        user_id,
        SUM(credits) as total_used,
        MAX(created_at) as last_usage
    FROM credit_usage 
    GROUP BY user_id
) usage ON u.id = usage.user_id;

-- =============================================
-- INITIAL SETUP COMPLETE
-- =============================================

-- Grant necessary permissions to service role
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- Comments
COMMENT ON TABLE credit_users IS 'User accounts for the AI Audio Services credit system';
COMMENT ON TABLE credit_purchases IS 'Record of all credit purchases via Stripe';
COMMENT ON TABLE credit_usage IS 'Record of all credit usage for AI services';

COMMENT ON FUNCTION add_user_credits IS 'Atomically add credits to user account from purchase';
COMMENT ON FUNCTION deduct_user_credits IS 'Atomically deduct credits for service usage';
COMMENT ON FUNCTION get_user_credit_summary IS 'Get comprehensive credit summary for user';

-- Schema version for tracking (optional)
-- CREATE TABLE IF NOT EXISTS schema_migrations (
--     version TEXT PRIMARY KEY,
--     name TEXT,
--     applied_at TIMESTAMPTZ DEFAULT NOW()
-- );
-- 
-- INSERT INTO schema_migrations (version, name) 
-- VALUES ('20250723_001', 'Klein Digital Solutions Credit System')
-- ON CONFLICT (version) DO NOTHING;

-- Schema successfully created
COMMENT ON SCHEMA public IS 'Klein Digital Solutions Credit System - Deployed on 2025-07-23';