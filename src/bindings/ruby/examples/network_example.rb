#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative '../lib/nimcp'

# Initialize NIMCP
NIMCP.init
puts "NIMCP version: #{NIMCP.version}"

# Create a neural network
network = NIMCP::Network.new(
  num_inputs: 10,
  num_outputs: 5,
  num_hidden: 20,
  learning_rate: 0.01
)

puts "\nCreated neural network:"
puts "  Inputs: 10"
puts "  Hidden: 20"
puts "  Outputs: #{network.num_outputs}"
puts "  Learning rate: 0.01"

puts "\n=== Forward Pass Tests ==="

# Test 1: All zeros
inputs = Array.new(10, 0.0)
outputs = network.forward(inputs)
puts "\nTest 1 - All zeros:"
puts "  Inputs:  #{inputs.inspect}"
puts "  Outputs: #{outputs.map { |v| v.round(4) }.inspect}"

# Test 2: All ones
inputs = Array.new(10, 1.0)
outputs = network.forward(inputs)
puts "\nTest 2 - All ones:"
puts "  Inputs:  #{inputs.inspect}"
puts "  Outputs: #{outputs.map { |v| v.round(4) }.inspect}"

# Test 3: Sequential values
inputs = (1..10).map(&:to_f)
outputs = network.forward(inputs)
puts "\nTest 3 - Sequential (1.0 to 10.0):"
puts "  Inputs:  #{inputs.inspect}"
puts "  Outputs: #{outputs.map { |v| v.round(4) }.inspect}"

# Test 4: Random values
inputs = Array.new(10) { rand }
outputs = network.forward(inputs)
puts "\nTest 4 - Random values:"
puts "  Inputs:  #{inputs.map { |v| v.round(4) }.inspect}"
puts "  Outputs: #{outputs.map { |v| v.round(4) }.inspect}"

# Test 5: Normalized pattern
inputs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
outputs = network.forward(inputs)
puts "\nTest 5 - Normalized pattern:"
puts "  Inputs:  #{inputs.inspect}"
puts "  Outputs: #{outputs.map { |v| v.round(4) }.inspect}"

# Cleanup
network.destroy
NIMCP.shutdown

puts "\n=== Done! ==="
