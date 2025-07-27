#!/bin/bash

# Individual Tool Testing Commands
# Use these to test specific tools one by one

export OPENAI_API_KEY="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"
export SERPER_API_KEY="84499b64573627a839be1208e5c6665132e635b0"

echo "🎯 Individual Tool Testing Commands"
echo "==================================="
echo ""

# Function to run a single test
run_test() {
    local test_name="$1"
    local description="$2" 
    local command="$3"
    
    echo "🔍 Test: $test_name"
    echo "Description: $description"
    echo "Command: $command"
    echo ""
    
    read -p "Run this test? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running test..."
        eval "$command"
        echo ""
        echo "✅ Test completed. Press Enter to continue..."
        read
    fi
    echo "----------------------------------------"
    echo ""
}

# Web Search Tool Test
run_test "WEB_SEARCH" \
    "Test SerperDev web search functionality" \
    "crewmaster create 'Search for latest developments in quantum computing and provide a summary' --name web_search_tool_test"

# Document Search Tool Test  
run_test "DOCUMENT_SEARCH" \
    "Test PDF, CSV, TXT document searching" \
    "crewmaster create 'Search through academic papers and documents to find information about neural networks' --name doc_search_tool_test"

# File Operations Tool Test
run_test "FILE_OPERATIONS" \
    "Test file reading capabilities" \
    "crewmaster create 'Read configuration files and extract important settings' --name file_ops_tool_test"

# Code Execution Tool Test
run_test "CODE_EXECUTION" \
    "Test Python code execution and file writing" \
    "crewmaster create 'Write Python code to analyze data and save results to CSV file' --name code_exec_tool_test"

# YouTube Search Tool Test
run_test "YOUTUBE_SEARCH" \
    "Test YouTube video and channel search" \
    "crewmaster create 'Find YouTube videos about artificial intelligence tutorials and summarize the top results' --name youtube_tool_test"

# Vision Tool Test
run_test "VISION_TOOL" \
    "Test image analysis and generation capabilities" \
    "crewmaster create 'Analyze images to identify objects and generate new images based on descriptions' --name vision_tool_test"

# Web Scraping Tool Test
run_test "WEB_SCRAPING" \
    "Test website scraping functionality" \
    "crewmaster create 'Scrape a news website to extract article headlines and summaries' --name scraping_tool_test"

# GitHub Search Tool Test (needs GITHUB_TOKEN)
run_test "GITHUB_SEARCH" \
    "Test GitHub repository search (requires GITHUB_TOKEN)" \
    "crewmaster create 'Search GitHub for popular machine learning repositories and analyze their features' --name github_tool_test"

# Database Search Tool Test
run_test "DATABASE_SEARCH" \
    "Test PostgreSQL database querying" \
    "crewmaster create 'Query database for user analytics and generate performance reports' --name db_tool_test"

# Browser Automation Tool Test (needs BROWSERBASE_API_KEY)
run_test "BROWSER_AUTOMATION" \
    "Test browser automation (requires BROWSERBASE_API_KEY)" \
    "crewmaster create 'Automate web browser to fill forms and extract dynamic content' --name browser_tool_test"

# API Calls Tool Test
run_test "API_CALLS" \
    "Test HTTP API calls functionality" \
    "crewmaster create 'Make API calls to weather services and process the JSON responses' --name api_tool_test"

# Email Tool Test
run_test "EMAIL_TOOL" \
    "Test email sending capabilities" \
    "crewmaster create 'Send email notifications with analysis results and attachments' --name email_tool_test"

# Data Processing Tool Test
run_test "DATA_PROCESSING" \
    "Test data analysis and processing" \
    "crewmaster create 'Process large datasets to identify patterns and generate statistical reports' --name data_proc_tool_test"

# Scheduling Tool Test
run_test "SCHEDULING" \
    "Test task scheduling functionality" \
    "crewmaster create 'Schedule automated reports and manage recurring tasks' --name schedule_tool_test"

echo "🎉 All individual tests completed!"
echo ""

# Show created crews
echo "📋 Created Test Crews:"
crewmaster list crews | grep "_tool_test" || echo "No test crews found"

echo ""
echo "🔍 Next steps:"
echo "1. Run specific crews: crewmaster run [crew_name]"
echo "2. Inspect crew details: crewmaster inspect [crew_name]" 
echo "3. Check performance: crewmaster performance [crew_name]"
echo "4. View tool usage: crewmaster stats --scope tools"