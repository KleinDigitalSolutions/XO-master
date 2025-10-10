// Service Worker for Klein Digital Solutions Music Studio
// Provides offline support and performance caching

const CACHE_NAME = 'klein-music-studio-v1';
const STATIC_CACHE_URLS = [
    '/',
    '/index_optimized.html',
    '/styles_optimized.css',
    '/performance-optimizations.js',
    '/favicons/favicon.ico',
    '/favicons/apple-touch-icon.png',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Service Worker: Caching static assets');
                return cache.addAll(STATIC_CACHE_URLS);
            })
            .then(() => {
                console.log('Service Worker: Static assets cached');
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('Service Worker: Caching failed', error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((cacheName) => cacheName !== CACHE_NAME)
                    .map((cacheName) => {
                        console.log('Service Worker: Deleting old cache', cacheName);
                        return caches.delete(cacheName);
                    })
            );
        }).then(() => {
            console.log('Service Worker: Activated');
            return self.clients.claim();
        })
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    // Skip non-GET requests
    if (event.request.method !== 'GET') {
        return;
    }

    // Skip API calls and external domains
    if (event.request.url.includes('/api/') || 
        event.request.url.includes('modal.run') ||
        !event.request.url.startsWith(self.location.origin)) {
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then((cachedResponse) => {
                // Return cached version if available
                if (cachedResponse) {
                    return cachedResponse;
                }

                // Otherwise fetch from network
                return fetch(event.request)
                    .then((response) => {
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }

                        // Clone the response before caching
                        const responseToCache = response.clone();

                        caches.open(CACHE_NAME)
                            .then((cache) => {
                                cache.put(event.request, responseToCache);
                            });

                        return response;
                    })
                    .catch(() => {
                        // Return offline page or fallback
                        if (event.request.destination === 'document') {
                            return caches.match('/offline.html');
                        }
                    });
            })
    );
});

// Background sync for failed uploads
self.addEventListener('sync', (event) => {
    if (event.tag === 'background-upload') {
        event.waitUntil(retryFailedUploads());
    }
});

async function retryFailedUploads() {
    // Implementation for retrying failed uploads
    console.log('Service Worker: Retrying failed uploads');
}

// Push notifications (for future use)
self.addEventListener('push', (event) => {
    const options = {
        body: event.data ? event.data.text() : 'Your music processing is complete!',
        icon: '/favicons/android-chrome-192x192.png',
        badge: '/favicons/android-chrome-96x96.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Results',
                icon: '/images/checkmark.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/images/xmark.png'
            }
        ]
    };

    event.waitUntil(
        self.registration.showNotification('Klein Digital Solutions', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});