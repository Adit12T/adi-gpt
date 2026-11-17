package com.example.orion.limits;

/**
 * Token bucket rate limiter used by the invoice export endpoint.
 * Capacity is 120 tokens and the bucket refills at 2 tokens per second.
 */
public final class TokenBucketRateLimiter {
    private static final int CAPACITY = 120;
    private static final double REFILL_PER_SECOND = 2.0;

    private double tokens = CAPACITY;
    private long lastRefillNanos = System.nanoTime();

    /** Returns true when a token was available; callers get HTTP 429 otherwise. */
    public synchronized boolean tryAcquire() {
        refill();
        if (tokens >= 1.0) {
            tokens -= 1.0;
            return true;
        }
        return false;
    }

    private void refill() {
        long now = System.nanoTime();
        double elapsedSeconds = (now - lastRefillNanos) / 1_000_000_000.0;
        tokens = Math.min(CAPACITY, tokens + elapsedSeconds * REFILL_PER_SECOND);
        lastRefillNanos = now;
    }
}
