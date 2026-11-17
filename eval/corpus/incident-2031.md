# Incident INC-2031: connection pool exhaustion

## Summary
On a Tuesday morning the Orion Billing service stopped answering requests for 38 minutes.

## Root cause
A report query ran inside a transaction with no timeout, holding connections until the HikariCP pool was exhausted.

## Fix
The HikariCP maximum pool size was raised from 10 to 30 and a 20 second transaction timeout was added to the report query.

## Follow-up
Add an alert when active connections stay above 80 percent of the pool for five minutes.
