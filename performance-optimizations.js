/**
 * Performance Optimizations for Klein Digital Solutions Music Studio
 * LANDR/TuneCore inspired performance enhancements
 */

// Service Worker for caching (PWA support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    });
}

// Intersection Observer for lazy loading and animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const fadeInObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-visible');
            fadeInObserver.unobserve(entry.target);
        }
    });
}, observerOptions);

// Apply fade-in animation to elements
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.service-card-modern, .result-card, .upload-zone-modern').forEach(el => {
        el.classList.add('fade-in-hidden');
        fadeInObserver.observe(el);
    });
});

// Debounced resize handler for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const handleResize = debounce(() => {
    // Handle responsive adjustments
    const viewportWidth = window.innerWidth;
    document.documentElement.style.setProperty('--viewport-width', `${viewportWidth}px`);
}, 250);

window.addEventListener('resize', handleResize, { passive: true });

// Optimized scroll handler
let ticking = false;
const handleScroll = () => {
    if (!ticking) {
        requestAnimationFrame(() => {
            const scrollY = window.scrollY;
            const navbar = document.querySelector('.navbar');
            
            if (scrollY > 50) {
                navbar?.classList.remove('transparent');
            } else {
                navbar?.classList.add('transparent');
            }
            
            ticking = false;
        });
        ticking = true;
    }
};

window.addEventListener('scroll', handleScroll, { passive: true });

// Preload critical resources
function preloadResource(href, as, type = null) {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    if (type) link.type = type;
    document.head.appendChild(link);
}

// Critical CSS preloading
const criticalResources = [
    { href: 'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2', as: 'font', type: 'font/woff2' },
    { href: '/favicons/apple-touch-icon.png', as: 'image' }
];

criticalResources.forEach(resource => {
    preloadResource(resource.href, resource.as, resource.type);
});

// Image lazy loading with Intersection Observer
const imageObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            imageObserver.unobserve(img);
        }
    });
});

document.querySelectorAll('img[data-src]').forEach(img => {
    imageObserver.observe(img);
});

// Memory management for file uploads
class MemoryOptimizedUpload {
    constructor() {
        this.maxChunkSize = 5 * 1024 * 1024; // 5MB chunks
        this.activeUploads = new Map();
    }

    async uploadLargeFile(file, endpoint, onProgress) {
        if (file.size <= this.maxChunkSize) {
            return this.uploadSmallFile(file, endpoint, onProgress);
        }

        const uploadId = Date.now().toString();
        const totalChunks = Math.ceil(file.size / this.maxChunkSize);
        
        try {
            for (let i = 0; i < totalChunks; i++) {
                const start = i * this.maxChunkSize;
                const end = Math.min(start + this.maxChunkSize, file.size);
                const chunk = file.slice(start, end);
                
                await this.uploadChunk(chunk, uploadId, i, totalChunks, endpoint);
                
                if (onProgress) {
                    onProgress(Math.round(((i + 1) / totalChunks) * 100));
                }
            }
            
            return await this.finalizeUpload(uploadId, endpoint);
        } catch (error) {
            this.cleanupUpload(uploadId);
            throw error;
        }
    }

    async uploadSmallFile(file, endpoint, onProgress) {
        const formData = new FormData();
        formData.append('audio_file', file);
        
        const xhr = new XMLHttpRequest();
        
        return new Promise((resolve, reject) => {
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress(Math.round((e.loaded / e.total) * 100));
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`Upload failed: ${xhr.status}`));
                }
            });
            
            xhr.addEventListener('error', () => reject(new Error('Upload failed')));
            xhr.open('POST', endpoint);
            xhr.send(formData);
        });
    }

    cleanupUpload(uploadId) {
        this.activeUploads.delete(uploadId);
    }
}

// Audio preview with Web Audio API
class AudioPreview {
    constructor() {
        this.audioContext = null;
        this.activeBuffers = new Map();
    }

    async initAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    async previewMIDI(midiData) {
        // Simplified MIDI preview using Web Audio API
        await this.initAudioContext();
        
        // This would implement basic MIDI playback
        // For production, you'd use a library like Tone.js
        console.log('MIDI preview functionality would be implemented here');
    }

    cleanup() {
        this.activeBuffers.clear();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
}

// Global instances
window.memoryOptimizedUpload = new MemoryOptimizedUpload();
window.audioPreview = new AudioPreview();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    window.audioPreview?.cleanup();
});

// Performance monitoring
if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
            if (entry.entryType === 'largest-contentful-paint') {
                console.log('LCP:', entry.startTime);
            }
            if (entry.entryType === 'first-input') {
                console.log('FID:', entry.processingStart - entry.startTime);
            }
        });
    });
    
    observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input'] });
}

// CSS animation utilities
const animationUtils = {
    fadeIn: (element, duration = 300) => {
        element.style.opacity = '0';
        element.style.transition = `opacity ${duration}ms ease-in-out`;
        element.style.display = 'block';
        
        requestAnimationFrame(() => {
            element.style.opacity = '1';
        });
    },
    
    fadeOut: (element, duration = 300) => {
        element.style.transition = `opacity ${duration}ms ease-in-out`;
        element.style.opacity = '0';
        
        setTimeout(() => {
            element.style.display = 'none';
        }, duration);
    },
    
    slideUp: (element, duration = 300) => {
        element.style.maxHeight = element.scrollHeight + 'px';
        element.style.transition = `max-height ${duration}ms ease-out`;
        element.style.overflow = 'hidden';
        
        requestAnimationFrame(() => {
            element.style.maxHeight = '0';
        });
    },
    
    slideDown: (element, duration = 300) => {
        element.style.maxHeight = '0';
        element.style.transition = `max-height ${duration}ms ease-out`;
        element.style.overflow = 'hidden';
        element.style.display = 'block';
        
        requestAnimationFrame(() => {
            element.style.maxHeight = element.scrollHeight + 'px';
        });
    }
};

// Export for use in main application
window.animationUtils = animationUtils;

// Add performance optimized CSS animations
const performanceCSS = `
/* Fade-in animations */
.fade-in-hidden {
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.6s ease-out, transform 0.6s ease-out;
}

.fade-in-visible {
    opacity: 1;
    transform: translateY(0);
}

/* Hardware accelerated animations */
.will-change-transform {
    will-change: transform;
}

.will-change-auto {
    will-change: auto;
}

/* Optimized hover effects */
@media (hover: hover) {
    .service-card-modern:hover {
        transform: translateY(-8px) translateZ(0);
    }
    
    .btn-modern:hover {
        transform: translateY(-2px) translateZ(0);
    }
    
    .result-card:hover {
        transform: translateY(-2px) translateZ(0);
    }
}

/* Reduce motion for accessibility */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Loading skeleton animations */
.skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Optimized focus styles */
.focus-visible {
    outline: 2px solid #667eea;
    outline-offset: 2px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    :root {
        --primary-gradient: linear-gradient(135deg, #000 0%, #333 100%);
        --text-primary: #000;
        --surface: #fff;
    }
}
`;

// Inject performance CSS
const styleSheet = document.createElement('style');
styleSheet.textContent = performanceCSS;
document.head.appendChild(styleSheet);