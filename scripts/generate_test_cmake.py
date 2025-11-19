#!/usr/bin/env python3
"""
Generate CMakeLists.txt files for all middleware tests
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "test"


def generate_cmake_for_directory(test_dir, test_type):
    """Generate CMakeLists.txt for a test directory"""

    # Find all .cpp files
    cpp_files = list(test_dir.glob("*.cpp"))
    if not cpp_files:
        return

    # Get category from path
    parts = test_dir.relative_to(TEST_DIR).parts
    if len(parts) >= 3:
        category = parts[2]  # e.g., "buffering", "normalization"
    else:
        category = "middleware"

    cmake_content = f"# Middleware {category.capitalize()} {test_type.capitalize()} Tests\n\n"

    for cpp_file in sorted(cpp_files):
        test_name = cpp_file.stem  # Remove .cpp extension
        executable_name = f"{test_type}_middleware_{category}_{test_name}"

        cmake_content += f"""# {test_name.replace('_', ' ').title()}
add_executable({executable_name}
    {cpp_file.name}
)
target_link_libraries({executable_name}
    PRIVATE
        gtest
        gtest_main
        nimcp_middleware
        nimcp_utils_memory
        nimcp_utils_platform
        nimcp_utils_time
        nimcp_utils_logging
)
add_test(NAME {executable_name}
         COMMAND {executable_name})

"""

    # Write CMakeLists.txt
    cmake_path = test_dir / "CMakeLists.txt"
    cmake_path.write_text(cmake_content)
    print(f"✓ Generated CMakeLists.txt: {test_dir.relative_to(PROJECT_ROOT)}")


def update_parent_cmake(parent_dir, subdirs):
    """Update parent CMakeLists.txt to include subdirectories"""

    cmake_content = f"# Middleware Tests\n\n"

    for subdir in sorted(subdirs):
        cmake_content += f"add_subdirectory({subdir})\n"

    cmake_path = parent_dir / "CMakeLists.txt"
    cmake_path.write_text(cmake_content)
    print(f"✓ Updated parent CMakeLists.txt: {parent_dir.relative_to(PROJECT_ROOT)}")


def main():
    """Generate all CMakeLists.txt files for middleware tests"""

    print("=== Middleware Test CMake Generator ===")
    print("")

    # Process unit tests
    unit_middleware_dir = TEST_DIR / "unit" / "middleware"
    unit_subdirs = []

    for category_dir in unit_middleware_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            generate_cmake_for_directory(category_dir, "unit")
            unit_subdirs.append(category_dir.name)

    # Handle unit/middleware/test_middleware_integration.cpp
    if (unit_middleware_dir / "test_middleware_integration.cpp").exists():
        cmake_content = """# Middleware Integration Tests

add_executable(unit_middleware_test_middleware_integration
    test_middleware_integration.cpp
)
target_link_libraries(unit_middleware_test_middleware_integration
    PRIVATE
        gtest
        gtest_main
        nimcp_middleware
        nimcp_utils_memory
)
add_test(NAME unit_middleware_test_middleware_integration
         COMMAND unit_middleware_test_middleware_integration)

"""
        for subdir in sorted(unit_subdirs):
            cmake_content += f"add_subdirectory({subdir})\n"

        (unit_middleware_dir / "CMakeLists.txt").write_text(cmake_content)
        print(f"✓ Generated CMakeLists.txt: test/unit/middleware")

    # Process integration tests
    integration_middleware_dir = TEST_DIR / "integration" / "middleware"
    if integration_middleware_dir.exists():
        generate_cmake_for_directory(integration_middleware_dir, "integration")

    # Process regression tests
    regression_middleware_dir = TEST_DIR / "regression" / "middleware"
    regression_subdirs = []

    if regression_middleware_dir.exists():
        for category_dir in regression_middleware_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                generate_cmake_for_directory(category_dir, "regression")
                regression_subdirs.append(category_dir.name)

        if regression_subdirs:
            update_parent_cmake(regression_middleware_dir, regression_subdirs)

    print("")
    print("✓ CMake generation complete!")
    print("")
    print("Next steps:")
    print("1. Review generated CMakeLists.txt files")
    print("2. Add middleware test directory to main test/CMakeLists.txt")
    print("3. Run: cd build && cmake .. && make")
    print("4. Run: ctest -R middleware")


if __name__ == "__main__":
    main()
