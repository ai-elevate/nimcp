# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = 'nimcp'
  spec.version       = '2.6.1'
  spec.authors       = ['NIMCP Contributors']
  spec.email         = ['noreply@nimcp.org']

  spec.summary       = 'Ruby bindings for NIMCP - Neural Interface Message Communication Protocol'
  spec.description   = 'NIMCP provides a cognitive computing framework with neural networks, ethics, and knowledge systems. This gem provides Ruby bindings via FFI.'
  spec.homepage      = 'https://github.com/nimcp/nimcp'
  spec.license       = 'MIT'
  spec.required_ruby_version = '>= 2.7.0'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = 'https://github.com/nimcp/nimcp'
  spec.metadata['changelog_uri'] = 'https://github.com/nimcp/nimcp/blob/master/CHANGELOG.md'

  spec.files = Dir['lib/**/*.rb', 'README.md', 'LICENSE']
  spec.require_paths = ['lib']

  # Dependencies
  spec.add_dependency 'ffi', '~> 1.15'

  # Development dependencies
  spec.add_development_dependency 'bundler', '~> 2.0'
  spec.add_development_dependency 'rake', '~> 13.0'
  spec.add_development_dependency 'rspec', '~> 3.0'
end
