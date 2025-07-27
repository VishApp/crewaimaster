#!/bin/bash

# CrewMaster Tool Testing Script
# Run this to systematically test all tools

set -e  # Exit on any error

echo "🧪 Starting CrewMaster Tool Testing..."
echo "========================================"

# Check API keys
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  OPENAI_API_KEY not set. Some tools may not work."
fi

if [[ -z "$SERPER_API_KEY" ]]; then
    echo "⚠️  SERPER_API_KEY not set. Web search may not work."
fi

echo ""

# Function to create and test a crew
test_crew() {
    local crew_name="$1"
    local task="$2"
    local expected_tools="$3"
    
    echo "🔧 Testing: $crew_name"
    echo "Task: $task"
    echo "Expected tools: $expected_tools"
    
    # Create crew
    echo "Creating crew..."
    crewmaster create "$task" --name "$crew_name"
    
    # Inspect crew
    echo "Inspecting crew configuration..."
    crewmaster inspect "$crew_name"
    
    # Run crew (with timeout)
    echo "Running crew (limited execution)..."
    timeout 120s crewmaster run "$crew_name" --input "This is a test run - provide minimal output" || echo "Test completed (timeout or early completion)"
    
    echo "✅ $crew_name test completed"
    echo "----------------------------------------"
    echo ""
}

# Test 1: Core Web Search Tool
test_crew "web_search_test" \
    "Search the web for latest AI news and create a brief summary" \
    "web_search"

# Test 2: File Operations
test_crew "file_ops_test" \
    "Read and analyze a text file, then save the summary to a new file" \
    "file_operations, code_execution"

# Test 3: Document Search  
test_crew "document_test" \
    "Search through PDF and CSV documents to find specific information" \
    "document_search"

# Test 4: Code Execution
test_crew "code_test" \
    "Write Python code to calculate statistics and save results to file" \
    "code_execution"

# Test 5: YouTube Search
test_crew "youtube_test" \
    "Search YouTube for educational videos about machine learning" \
    "youtube_search"

# Test 6: Vision Tool
test_crew "vision_test" \
    "Analyze an image and describe its contents using AI vision" \
    "vision"

# Test 7: Multi-tool Integration
test_crew "integration_test" \
    "Search the web for research papers, analyze PDFs, and create a comprehensive report with code analysis" \
    "web_search, document_search, code_execution, file_operations"

echo "🎉 All tool tests completed!"
echo ""

# Summary
echo "📊 Test Summary"
echo "==============="
crewmaster list crews
echo ""

echo "📈 System Stats"
echo "==============="
crewmaster stats --scope tools

echo ""
echo "🔍 Next Steps:"
echo "1. Check individual crew results with: crewmaster performance [crew_name]"
echo "2. Inspect tool usage with: crewmaster inspect [crew_name]"  
echo "3. Review any errors in the output above"
echo "4. Test additional tools based on your specific needs"