# Build stage
FROM rust:1.75-slim as builder

WORKDIR /usr/src/app
COPY . .

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/app/target/release/vmsam-web /app/vmsam-web
COPY static /app/static

CMD ["./vmsam-web"]
