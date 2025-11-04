#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative '../lib/nimcp'

# Initialize NIMCP
NIMCP.init
puts "NIMCP version: #{NIMCP.version}"

# Create a brain for classification
brain = NIMCP::Brain.new(
  name: 'fruit_classifier',
  size: :small,
  task: :classification,
  num_inputs: 3,  # RGB color values
  num_outputs: 3  # apple, banana, orange
)

puts "\n=== Training Phase ==="

# Train with apple samples (red)
brain.learn([0.9, 0.1, 0.1], 'apple', 0.95)
brain.learn([0.8, 0.2, 0.1], 'apple', 0.90)
brain.learn([0.95, 0.05, 0.05], 'apple', 0.98)

# Train with banana samples (yellow)
brain.learn([0.9, 0.9, 0.1], 'banana', 0.95)
brain.learn([0.85, 0.85, 0.15], 'banana', 0.90)
brain.learn([0.95, 0.95, 0.05], 'banana', 0.92)

# Train with orange samples (orange)
brain.learn([0.9, 0.5, 0.1], 'orange', 0.95)
brain.learn([0.85, 0.55, 0.15], 'orange', 0.93)
brain.learn([0.95, 0.45, 0.05], 'orange', 0.91)

puts "Trained on 9 samples (3 apples, 3 bananas, 3 oranges)"

puts "\n=== Prediction Phase ==="

# Test predictions
test_samples = [
  { color: [0.92, 0.08, 0.08], expected: 'apple' },
  { color: [0.88, 0.88, 0.12], expected: 'banana' },
  { color: [0.87, 0.52, 0.13], expected: 'orange' },
  { color: [0.90, 0.90, 0.08], expected: 'banana' }
]

test_samples.each_with_index do |sample, i|
  result = brain.predict(sample[:color])
  rgb = sample[:color].map { |v| (v * 255).round }.join(', ')

  puts "\nTest #{i + 1}:"
  puts "  RGB: (#{rgb})"
  puts "  Expected: #{sample[:expected]}"
  puts "  Predicted: #{result[:label]}"
  puts "  Confidence: #{(result[:confidence] * 100).round(2)}%"
  puts "  ✓ Correct!" if result[:label] == sample[:expected]
end

puts "\n=== Save/Load Test ==="

# Save the brain
filepath = '/tmp/fruit_classifier.brain'
brain.save(filepath)
puts "Brain saved to: #{filepath}"

# Load it back
loaded_brain = NIMCP::Brain.load(filepath)
puts "Brain loaded successfully"

# Verify it works
result = loaded_brain.predict([0.9, 0.1, 0.1])
puts "Prediction from loaded brain: #{result[:label]} (confidence: #{(result[:confidence] * 100).round(2)}%)"

# Cleanup
brain.destroy
loaded_brain.destroy
NIMCP.shutdown

puts "\n=== Done! ==="
