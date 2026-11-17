# API conventions

## Errors
Error codes use the ORN prefix. ORN-4041 means the customer was not found and ORN-4092 means a duplicate invoice.

## Pagination
List endpoints default to a page size of 50 and allow at most 200.

## Tracing
Every request must carry the X-Orion-Trace header; the gateway adds it when the caller does not send one.

## Versioning
Breaking changes go under a new path prefix such as /v2 and the old version stays for six months.
