# Orion Billing service

## Overview
Orion Billing is a Spring Boot service that creates invoices for the claims platform. It is owned by team Kestrel.

## Runtime
The service listens on port 8431. It stores invoices in PostgreSQL 16 and uses Flyway for schema migrations.

## Retry policy
Calls to the payment gateway are retried three times with exponential backoff, starting at 250 ms. After the third failure the invoice is parked in the `invoice_dead_letter` table.

## Scheduled jobs
The nightly reconciliation job runs at 02:30 UTC and is named `ReconcileLedgerJob`.
