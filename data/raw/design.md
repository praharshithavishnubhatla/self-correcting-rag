File Name: design_url_shortener.md

# System Design: URL Shortener

## Functional Requirements
- Shorten long URLs
- Redirect users quickly
- Track click analytics

## Non-Functional Requirements
- 99.99% availability
- Sub-100ms latency
- Highly scalable

---

## API Design

POST /shorten
Request:
{
  "long_url": "https://example.com"
}

Response:
{
  "short_url": "https://sho.rt/abc123"
}

---

## Capacity Estimation
Assume:
- 100M URLs per month
- 100 redirects per second

Storage:
If each URL takes 500 bytes:
100M * 500B = 50GB/month

---

## Database Choice
Use NoSQL (e.g., DynamoDB, Cassandra) because:
- Massive scale
- Fast lookups
- Flexible schema

---

## ID Generation Strategies
- Base62 encoding
- Snowflake IDs
- Hashing + collision detection

---

## High-Level Architecture

Client → Load Balancer → App Servers → Cache → Database

---

## Bottlenecks to Watch
- Hot keys
- Database overload
- Cache misses

---

## Optimizations
- Use Redis for redirects
- Enable CDN caching
- Bloom filters to prevent DB hits for invalid keys
