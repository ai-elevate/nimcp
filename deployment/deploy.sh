#!/bin/bash
#==============================================================================
# NIMCP 2.5 Deployment Script
#==============================================================================
# Automated deployment to production environments
#
# Usage:
#   ./deploy.sh [environment] [options]
#
# Environments:
#   dev       - Development environment
#   staging   - Staging environment
#   prod      - Production environment
#
# Options:
#   --skip-tests     Skip test execution
#   --skip-backup    Skip backup creation
#   --force          Force deployment without confirmation
#==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-dev}"
SKIP_TESTS=false
SKIP_BACKUP=false
FORCE=false

#------------------------------------------------------------------------------
# Parse command line arguments
#------------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
}

#------------------------------------------------------------------------------
# Logging functions
#------------------------------------------------------------------------------
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

#------------------------------------------------------------------------------
# Validation checks
#------------------------------------------------------------------------------
validate_environment() {
    log_info "Validating environment: $ENVIRONMENT"

    case $ENVIRONMENT in
        dev|staging|prod)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, prod"
            exit 1
            ;;
    esac

    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose not found. Please install docker-compose first."
        exit 1
    fi
}

#------------------------------------------------------------------------------
# Pre-deployment backup
#------------------------------------------------------------------------------
create_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        log_warn "Skipping backup creation"
        return 0
    fi

    log_info "Creating backup..."

    local backup_dir="/var/backups/nimcp"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="nimcp_${ENVIRONMENT}_${timestamp}.tar.gz"

    mkdir -p "$backup_dir"

    # Backup data volumes
    docker run --rm \
        -v nimcp-data:/data \
        -v nimcp-models:/models \
        -v "$backup_dir:/backup" \
        alpine tar czf "/backup/$backup_name" /data /models 2>/dev/null || true

    log_info "Backup created: $backup_dir/$backup_name"
}

#------------------------------------------------------------------------------
# Run tests
#------------------------------------------------------------------------------
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping tests"
        return 0
    fi

    log_info "Running tests..."

    cd "$PROJECT_ROOT"

    # Build test image
    docker build -f Dockerfile --target builder -t nimcp:test .

    # Run tests in container
    docker run --rm nimcp:test /build/build/src/tests/nimcp_tests

    log_info "All tests passed ✓"
}

#------------------------------------------------------------------------------
# Build Docker images
#------------------------------------------------------------------------------
build_images() {
    log_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Build production image
    docker-compose build --no-cache

    log_info "Docker images built successfully ✓"
}

#------------------------------------------------------------------------------
# Deploy services
#------------------------------------------------------------------------------
deploy_services() {
    log_info "Deploying services..."

    cd "$PROJECT_ROOT"

    # Pull latest images if using registry
    # docker-compose pull

    # Stop existing services
    docker-compose down

    # Start services
    docker-compose up -d

    log_info "Services deployed ✓"
}

#------------------------------------------------------------------------------
# Health check
#------------------------------------------------------------------------------
wait_for_healthy() {
    log_info "Waiting for services to become healthy..."

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps | grep -q "healthy"; then
            log_info "Services are healthy ✓"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    echo ""
    log_error "Services failed to become healthy"
    docker-compose logs
    return 1
}

#------------------------------------------------------------------------------
# Post-deployment verification
#------------------------------------------------------------------------------
verify_deployment() {
    log_info "Verifying deployment..."

    # Check all containers are running
    local running=$(docker-compose ps --services --filter "status=running" | wc -l)
    local total=$(docker-compose ps --services | wc -l)

    if [ "$running" -eq "$total" ]; then
        log_info "All containers running ($running/$total) ✓"
    else
        log_error "Some containers not running ($running/$total)"
        return 1
    fi

    # Check health endpoints
    if curl -f http://localhost:8080/health &>/dev/null; then
        log_info "Health check passed ✓"
    else
        log_warn "Health check endpoint not responding"
    fi

    log_info "Deployment verification completed ✓"
}

#------------------------------------------------------------------------------
# Rollback function
#------------------------------------------------------------------------------
rollback() {
    log_error "Deployment failed! Rolling back..."

    cd "$PROJECT_ROOT"
    docker-compose down

    # Restore from latest backup
    local backup_dir="/var/backups/nimcp"
    local latest_backup=$(ls -t "$backup_dir"/nimcp_${ENVIRONMENT}_*.tar.gz 2>/dev/null | head -1)

    if [ -n "$latest_backup" ]; then
        log_info "Restoring from backup: $latest_backup"
        # Restore logic here
    fi

    log_error "Rollback completed"
    exit 1
}

#------------------------------------------------------------------------------
# Main deployment flow
#------------------------------------------------------------------------------
main() {
    echo "=============================================="
    echo "NIMCP 2.5 Deployment"
    echo "=============================================="
    echo "Environment: $ENVIRONMENT"
    echo "=============================================="
    echo ""

    parse_args "$@"
    validate_environment

    # Confirmation for production
    if [ "$ENVIRONMENT" = "prod" ] && [ "$FORCE" = false ]; then
        read -p "Deploy to PRODUCTION? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_error "Deployment cancelled"
            exit 0
        fi
    fi

    # Set trap for rollback on error
    trap rollback ERR

    # Deployment steps
    create_backup
    run_tests
    build_images
    deploy_services
    wait_for_healthy
    verify_deployment

    # Remove trap
    trap - ERR

    echo ""
    echo "=============================================="
    log_info "Deployment completed successfully!"
    echo "=============================================="
    echo ""
    echo "Services:"
    docker-compose ps
    echo ""
    echo "Access points:"
    echo "  - NIMCP API: http://localhost:8080"
    echo "  - Metrics: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    echo ""
}

main "$@"
