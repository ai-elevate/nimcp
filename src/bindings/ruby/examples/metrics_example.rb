#!/usr/bin/env ruby
# frozen_string_literal: true

#
# NIMCP Metrics Example
# Demonstrates metrics collection for Tableau and PowerBI
#

require_relative '../lib/nimcp'

begin
  # Initialize NIMCP
  NIMCP.init
  puts "NIMCP Version: #{NIMCP.version}"

  # Create metrics collector
  metrics = NIMCP::MetricsCollector.new(
    directory: './ruby_metrics',
    format: :csv
  )
  puts "\nMetrics collector created"

  # Record various metrics
  puts "\nRecording metrics..."

  # Counter metrics
  metrics.record_counter('forward_passes', 1000, category: :performance)
  metrics.record_counter('training_examples', 5000, category: :learning)

  # Gauge metrics
  metrics.record_gauge('learning_rate', 0.001, category: :learning)
  metrics.record_gauge('accuracy', 0.95, category: :inference)
  metrics.record_gauge('memory_usage_mb', 256.5, category: :memory)

  # Timer metrics
  metrics.record_timer('forward_time_ms', 15.5, category: :performance)
  metrics.record_timer('backward_time_ms', 22.3, category: :performance)

  # Timing block example
  metrics.timer('data_processing', category: :performance) do
    # Simulate some work
    sleep(0.1)
    puts "Processing data..."
  end

  # Simulate training loop with metrics
  puts "\nSimulating training loop with metrics..."
  5.times do |epoch|
    metrics.timer("epoch_#{epoch}", category: :learning) do
      # Record epoch-specific metrics
      metrics.record_gauge('current_epoch', epoch, category: :learning)
      metrics.record_gauge('loss', 0.5 / (epoch + 1), category: :learning)

      # Simulate training time
      sleep(0.05)
    end
  end

  # Get and display statistics
  puts "\nMetrics Statistics:"
  stats = metrics.stats
  puts "  Total metrics recorded: #{stats['total_recorded']}"
  puts "  Total metrics flushed: #{stats['total_flushed']}"
  puts "  Buffer count: #{stats['buffer_count']}"

  # Flush metrics to disk
  puts "\nFlushing metrics..."
  count = metrics.flush
  puts "  Flushed #{count} metrics to disk"

  # Export for Tableau
  tableau_file = './ruby_metrics_tableau.csv'
  puts "\nExporting to Tableau CSV: #{tableau_file}"
  metrics.export_tableau_csv(tableau_file)

  # Export for PowerBI
  powerbi_file = './ruby_metrics_powerbi.json'
  puts "Exporting to PowerBI JSON: #{powerbi_file}"
  metrics.export_powerbi_json(powerbi_file)

  # Change directory and record more metrics
  puts "\nChanging metrics directory..."
  metrics.set_directory('./ruby_metrics_alternate')
  metrics.record_counter('additional_operations', 100, category: :system)

  # Get updated statistics
  stats = metrics.stats
  puts "\nFinal Statistics:"
  puts "  Total metrics recorded: #{stats['total_recorded']}"

  # Clean up
  metrics.destroy
  puts "\nMetrics collector destroyed"

  # Shutdown NIMCP
  NIMCP.shutdown
  puts "NIMCP shutdown complete"

rescue NIMCP::Error => e
  puts "Error: #{e.message}"
  exit 1
end
