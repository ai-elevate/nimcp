#==============================================================================
# NIMCP 2.5 Production Docker Image
#==============================================================================
# Multi-stage build for optimized production image
# Base: Ubuntu 22.04 with build tools → Runtime: Minimal Ubuntu
#==============================================================================

#------------------------------------------------------------------------------
# Stage 1: Builder - Compile NIMCP with all dependencies
#------------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

# Metadata
LABEL maintainer="NIMCP Development Team"
LABEL version="2.5.0"
LABEL description="Neural Inference for Massive Concurrent Processing with Golden Rule Ethics"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libpython3.10-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy source code
COPY . /build/

# Build NIMCP
RUN mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/nimcp && \
    make -j$(nproc) && \
    make install

# Run tests to verify build
RUN cd build && ./src/tests/nimcp_tests

#------------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
#------------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

# Metadata
LABEL maintainer="NIMCP Development Team"
LABEL version="2.5.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libpython3.10 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create nimcp user for security (non-root)
RUN useradd -r -u 1000 -m -s /bin/bash nimcp

# Copy built artifacts from builder
COPY --from=builder /opt/nimcp /opt/nimcp
COPY --from=builder /build/build/src/lib/*.so /usr/local/lib/
COPY --from=builder /build/build/lib/python/nimcp.so /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /build/build/examples/* /opt/nimcp/bin/

# Copy example data and configs
COPY --from=builder /build/examples /opt/nimcp/examples
COPY --from=builder /build/docs /opt/nimcp/docs

# Update library cache
RUN ldconfig

# Create necessary directories
RUN mkdir -p /var/log/nimcp /var/lib/nimcp /etc/nimcp && \
    chown -R nimcp:nimcp /var/log/nimcp /var/lib/nimcp /opt/nimcp

# Health check script
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Environment variables
ENV NIMCP_HOME=/opt/nimcp
ENV NIMCP_DATA=/var/lib/nimcp
ENV NIMCP_LOG=/var/log/nimcp
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/nimcp/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/nimcp/bin:$PATH

# Expose ports for monitoring and API (if applicable)
EXPOSE 8080 9090

# Switch to non-root user
USER nimcp
WORKDIR /opt/nimcp

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD /usr/local/bin/healthcheck.sh || exit 1

# Default command: Run integrated demo
CMD ["/opt/nimcp/bin/integrated_demo"]
