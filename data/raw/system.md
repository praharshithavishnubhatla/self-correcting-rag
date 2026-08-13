File Name: scalable_system_design.md

# Scalable System Design

## What is Scalability?
Scalability refers to a system's ability to handle growing workloads without performance degradation.

---

## Types of Scalability

### 1. Horizontal Scaling
Adding more machines to the system.

**Advantages:**
- Fault tolerant
- Cost efficient at scale
- Flexible

**Disadvantages:**
- Network complexity
- Requires distributed architecture

---

### 2. Vertical Scaling
Increasing CPU/RAM on a single server.

**Advantages:**
- Simpler architecture
- Easier to manage

**Disadvantages:**
- Hardware limits
- Expensive upgrades

---

## Load Balancing

Load balancers act as traffic controllers.

**Popular Algorithms:**
- Round Robin
- Least Connections
- IP Hash

**Common Tools:**
- Nginx
- HAProxy
- AWS ELB

---

## Caching Strategy

Caching improves latency dramatically.

**Where to Cache:**
- CDN
- Database query cache
- Application cache

**Cache Patterns:**
- Cache Aside
- Write Through
- Write Back

---

## Database Scaling

### Read Replicas
Used to scale read-heavy applications.

### Sharding
Splitting a database into smaller pieces called shards.

Example:
Users with IDs 1–1M → Shard 1  
Users with IDs 1M–2M → Shard 2  

---

## System Design Example: Twitter Feed

### Requirements
- Millions of users
- Real-time updates
- Low latency

### High-Level Design
1. Client sends request
2. Load balancer routes traffic
3. Application servers process requests
4. Feed generation service builds timeline
5. Cache stores recent feeds
6. Database persists tweets

---

## Key Takeaways
- Avoid single points of failure
- Design for scale early
- Use caching aggressively
- Prefer stateless services
