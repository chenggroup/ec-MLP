#!/bin/bash

# Script to find and run all run_test.sh files in ec-MLP/tests/lmp directory
# with a progress bar

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display progress bar
show_progress() {
	local current=$1
	local total=$2
	local width=50
	local percentage=$((current * 100 / total))
	local filled=$((current * width / total))
	local empty=$((width - filled))

	printf "\r${BLUE}Progress: ["
	printf "%*s" $filled | tr ' ' '='
	printf "%*s" $empty | tr ' ' '-'
	printf "] %d%% (%d/%d)${NC}" $percentage $current $total
}

# Function to run a single test
run_test() {
	local test_file=$1
	local test_name=$(basename $(dirname $test_file))

	echo -e "\n${YELLOW}Running test: $test_name${NC}"
	echo -e "${BLUE}Executing: $test_file${NC}"

	# Run the test and capture output
	if bash "$test_file" >/dev/null 2>&1; then
		echo -e "${GREEN}✓ Test passed: $test_name${NC}"
		return 0
	else
		echo -e "${RED}✗ Test failed: $test_name${NC}"
		return 1
	fi
}

# Main script
main() {
	echo -e "${BLUE}Searching for run_test.sh files in ec-MLP/tests/lmp...${NC}"

	# Find all run_test.sh files
	test_files=()
	while IFS= read -r -d '' file; do
		test_files+=("$file")
	done < <(find "$(dirname "$0")" -name "run_test.sh" -type f -print0)

	if [ ${#test_files[@]} -eq 0 ]; then
		echo -e "${RED}No run_test.sh files found!${NC}"
		exit 1
	fi

	echo -e "${GREEN}Found ${#test_files[@]} test file(s)${NC}"
	echo

	# Run tests with progress bar
	total_tests=${#test_files[@]}
	passed_tests=0
	failed_tests=0

	for i in "${!test_files[@]}"; do
		current_test=$((i + 1))
		test_file="${test_files[$i]}"

		show_progress $current_test $total_tests

		# Run the test
		if run_test "$test_file"; then
			((passed_tests++))
		else
			((failed_tests++))
		fi

		# Small delay for better visibility
		sleep 0.5
	done

	# Final progress bar at 100%
	show_progress $total_tests $total_tests
	echo -e "\n"

	# Summary
	echo -e "${BLUE}=== Test Summary ===${NC}"
	echo -e "${GREEN}Passed: $passed_tests${NC}"
	echo -e "${RED}Failed: $failed_tests${NC}"
	echo -e "${BLUE}Total:  $total_tests${NC}"

	if [ $failed_tests -eq 0 ]; then
		echo -e "\n${GREEN}All tests passed successfully!${NC}"
		exit 0
	else
		echo -e "\n${RED}Some tests failed. Please check the output above.${NC}"
		exit 1
	fi
}

# Run the main function
main "$@"
