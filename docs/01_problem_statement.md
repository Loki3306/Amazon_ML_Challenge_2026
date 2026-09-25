# Problem Statement

The challenge is a **Business Entity Resolution** problem. Business identity data arrives from three independent sources with noisy and inconsistent representations.

- **Source 1** is the deduplicated reference source.
- We must map each Source 1 entity to zero or more entities in **Source 2** and **Source 3**.
- This is a one-to-many entity-resolution problem requiring explicit candidate-generation/blocking.
