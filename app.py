import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from database import DatabaseManager
from document_processor import DocumentProcessor
from web_scraper import get_website_text_content
from azure_openai_client import AzureOpenAIClient
from evaluation_engine import EvaluationEngine
from utils import generate_summary_stats, format_timestamp

# Page configuration
st.set_page_config(
    page_title="RankRight - Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    db_manager = DatabaseManager()
    doc_processor = DocumentProcessor()
    ai_client = AzureOpenAIClient()
    eval_engine = EvaluationEngine(ai_client)
    return db_manager, doc_processor, ai_client, eval_engine

db_manager, doc_processor, ai_client, eval_engine = init_components()

# Session state initialization
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_analysis_id' not in st.session_state:
    st.session_state.current_analysis_id = None

def main():
    st.title("🎯 RankRight - Intelligent Document Analyzer")
    st.markdown("Analyze documents and Confluence pages with AI-powered insights")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", ["Home", "Analysis History", "Settings"])
    
    if page == "Home":
        show_home_page()
    elif page == "Analysis History":
        show_history_page()
    elif page == "Settings":
        show_settings_page()

def show_home_page():
    st.header("Document Analysis")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["Upload Documents", "Confluence URL"])
    
    content_text = ""
    source_info = ""
    
    if input_method == "Upload Documents":
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            content_text = ""
            source_info = f"Uploaded files: {', '.join([f.name for f in uploaded_files])}"
            
            for uploaded_file in uploaded_files:
                try:
                    file_content = doc_processor.process_file(uploaded_file)
                    content_text += f"\n\n--- Content from {uploaded_file.name} ---\n{file_content}"
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
    else:  # Confluence URL
        url = st.text_input("Enter Confluence page URL:")
        
        if url and st.button("Fetch Content"):
            try:
                with st.spinner("Fetching content from URL..."):
                    content_text = get_website_text_content(url)
                    source_info = f"URL: {url}"
                    st.success("Content fetched successfully!")
            except Exception as e:
                st.error(f"Error fetching content: {str(e)}")
    
    # Analysis section
    if content_text:
        st.subheader("Content Preview")
        with st.expander("View extracted content", expanded=False):
            st.text_area("Content", content_text[:1000] + "..." if len(content_text) > 1000 else content_text, height=200, disabled=True)
        
        # Analysis button
        if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
            perform_analysis(content_text, source_info)
    
    # Show analysis results if available
    if st.session_state.analysis_complete and st.session_state.current_analysis_id:
        show_analysis_results(st.session_state.current_analysis_id)

def perform_analysis(content_text, source_info):
    """Perform comprehensive document analysis"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Generate summary
        status_text.text("Generating document summary...")
        progress_bar.progress(20)
        
        summary = ai_client.summarize_content(content_text)
        
        # Step 2: Run evaluation criteria
        status_text.text("Evaluating against criteria...")
        progress_bar.progress(40)
        
        evaluation_results = eval_engine.evaluate_content(content_text)
        
        # Step 3: Store results in database
        status_text.text("Storing analysis results...")
        progress_bar.progress(80)
        
        analysis_id = db_manager.store_analysis(
            content=content_text,
            source_info=source_info,
            summary=summary,
            evaluation_results=evaluation_results
        )
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Analysis completed successfully!")
        
        # Update session state
        st.session_state.analysis_complete = True
        st.session_state.current_analysis_id = analysis_id
        
        st.success("✅ Analysis completed! Results are displayed below.")
        st.rerun()
        
    except Exception as e:
        error_msg = str(e)
        progress_bar.empty()
        status_text.empty()
        
        # Check for Azure firewall/networking issues
        if "403" in error_msg and ("Virtual Network" in error_msg or "Firewall" in error_msg):
            st.error("🚫 Azure OpenAI Access Blocked")
            
            with st.expander("🔧 Quick Fix Instructions", expanded=True):
                st.markdown("""
                **Your Azure OpenAI service has firewall rules blocking this connection.**
                
                ### Option 1: Allow All Networks (Simplest)
                1. Go to **Azure Portal** (portal.azure.com)
                2. Find your **Azure OpenAI resource**
                3. Click **"Networking"** in the left sidebar
                4. Change from "Selected networks" to **"All networks"**
                5. Click **"Save"** and wait 2-3 minutes
                
                ### Option 2: Add Current IP (Temporary)
                - Add IP: `34.9.104.227` to your allowed list
                - Note: This IP changes when Replit restarts
                
                ### Option 3: Check Virtual Network Settings
                - Your error suggests VNet is configured
                - Go to Networking → Public network access
                - Change to "Enabled from all networks"
                """)
                
            st.info("After fixing the Azure settings, try uploading and analyzing your document again.")
            
        else:
            st.error(f"Analysis failed: {error_msg}")

def show_analysis_results(analysis_id):
    """Display analysis results in tabs"""
    
    # Get analysis data
    analysis_data = db_manager.get_analysis(analysis_id)
    if not analysis_data:
        st.error("Analysis data not found")
        return
    
    # Summary section
    st.subheader("📄 Document Summary")
    st.write(analysis_data['summary'])
    
    # Evaluation results in tabs
    st.subheader("📊 Evaluation Results")
    
    # Create tabs for each criterion
    criteria_names = list(analysis_data['evaluation_results'].keys())
    tabs = st.tabs(criteria_names)
    
    for i, (criterion_name, tab) in enumerate(zip(criteria_names, tabs)):
        with tab:
            result = analysis_data['evaluation_results'][criterion_name]
            
            # Display ranking with color coding
            ranking = result['ranking']
            color_map = {'Green': '🟢', 'Amber': '🟡', 'Red': '🔴'}
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Ranking",
                    value=f"{color_map.get(ranking, '⚪')} {ranking}",
                )
                st.metric(
                    label="Score",
                    value=f"{result['score']:.1f}/10"
                )
            
            with col2:
                # Score visualization
                fig = px.bar(
                    x=['Score'],
                    y=[result['score']],
                    color_discrete_sequence=['#2E8B57' if ranking == 'Green' 
                                           else '#FF8C00' if ranking == 'Amber' 
                                           else '#DC143C'],
                    title=f"{criterion_name} Score"
                )
                fig.update_layout(height=200, showlegend=False)
                fig.update_yaxis(range=[0, 10])
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed explanation
            st.subheader("Analysis Details")
            st.write(result['explanation'])
            
            # Key findings
            if 'key_findings' in result:
                st.subheader("Key Findings")
                for finding in result['key_findings']:
                    st.write(f"• {finding}")
            
            # Recommendations
            if 'recommendations' in result:
                st.subheader("Recommendations")
                for recommendation in result['recommendations']:
                    st.write(f"• {recommendation}")

def show_history_page():
    """Display analysis history"""
    st.header("📚 Analysis History")
    
    # Get all analyses
    analyses = db_manager.get_all_analyses()
    
    if not analyses:
        st.info("No analyses found. Start by analyzing a document on the Home page.")
        return
    
    # Create summary table
    df_data = []
    for analysis in analyses:
        df_data.append({
            'ID': analysis['id'],
            'Source': analysis['source_info'][:50] + "..." if len(analysis['source_info']) > 50 else analysis['source_info'],
            'Timestamp': format_timestamp(analysis['timestamp']),
            'Summary': analysis['summary'][:100] + "..." if len(analysis['summary']) > 100 else analysis['summary']
        })
    
    df = pd.DataFrame(df_data)
    
    # Display table with selection
    selected_indices = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Show detailed view for selected analysis
    if selected_indices['selection']['rows']:
        selected_idx = selected_indices['selection']['rows'][0]
        selected_id = df.iloc[selected_idx]['ID']
        
        st.divider()
        st.subheader(f"Analysis Details - ID: {selected_id}")
        show_analysis_results(selected_id)

def show_settings_page():
    """Display settings and configuration"""
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    # Azure OpenAI settings
    with st.expander("Azure OpenAI Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Configuration**")
            endpoint = st.text_input("Azure OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
            api_key = st.text_input("API Key", type="password", value="***" if os.getenv("AZURE_OPENAI_API_KEY") else "")
            api_version = st.text_input("API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
            deployment_name = st.text_input("Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "RankRightAnalyzer"))
        
        with col2:
            st.markdown("**Private Endpoint Configuration**")
            use_private_endpoint = st.checkbox(
                "Use Private Endpoint", 
                value=os.getenv("AZURE_OPENAI_USE_PRIVATE_ENDPOINT", "false").lower() == "true",
                help="Enable to connect via private endpoint instead of public internet"
            )
            
            private_ip = st.text_input(
                "Private Endpoint IP", 
                value=os.getenv("AZURE_OPENAI_PRIVATE_IP", ""),
                disabled=not use_private_endpoint,
                help="IP address of the private endpoint"
            )
            
            private_fqdn = st.text_input(
                "Private Endpoint FQDN (Optional)", 
                value=os.getenv("AZURE_OPENAI_PRIVATE_FQDN", ""),
                disabled=not use_private_endpoint,
                help="Custom FQDN for private endpoint (leave empty to use IP directly)"
            )
        
        # Save configuration button
        if st.button("💾 Save Configuration", key="save_config"):
            try:
                from config_manager import ConfigManager
                config_mgr = ConfigManager()
                
                # Update configuration
                config_mgr.update_azure_openai_config(
                    endpoint=endpoint,
                    api_version=api_version,
                    deployment_name=deployment_name,
                    use_private_endpoint=use_private_endpoint,
                    private_endpoint_ip=private_ip,
                    private_endpoint_fqdn=private_fqdn
                )
                
                st.success("✅ Configuration saved successfully!")
                st.info("""
                **Next Steps:**
                1. Restart the application to apply changes
                2. Test the connection using the button below
                3. If using private endpoint, ensure network connectivity to the private IP
                """)
                
            except Exception as e:
                st.error(f"❌ Failed to save configuration: {str(e)}")
        
        # Configuration check
        st.subheader("Configuration Status")
        
        # Show connection information
        try:
            from azure_openai_client import AzureOpenAIClient
            client = AzureOpenAIClient()
            conn_info = client.get_connection_info()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Configuration**")
                if conn_info["endpoint"]:
                    st.info(f"✅ Endpoint: {conn_info['endpoint']}")
                else:
                    st.error("❌ Azure OpenAI Endpoint is missing")
                    
                if os.getenv("AZURE_OPENAI_API_KEY"):
                    st.info("✅ API Key is configured")
                else:
                    st.error("❌ Azure OpenAI API Key is missing")
                    
                if conn_info["deployment_name"]:
                    st.info(f"✅ Deployment: {conn_info['deployment_name']}")
                else:
                    st.error("❌ Deployment Name is missing")
            
            with col2:
                st.markdown("**Connection Information**")
                if conn_info["use_private_endpoint"]:
                    st.info("🔒 Private Endpoint: Enabled")
                    st.info(f"📍 Effective Endpoint: {conn_info['effective_endpoint']}")
                    if conn_info["private_endpoint_ip"]:
                        st.info(f"🌐 Private IP: {conn_info['private_endpoint_ip']}")
                    if conn_info["private_endpoint_fqdn"]:
                        st.info(f"🏷️ Private FQDN: {conn_info['private_endpoint_fqdn']}")
                else:
                    st.info("🌐 Private Endpoint: Disabled (Using public endpoint)")
                    
        except Exception as e:
            st.error(f"❌ Configuration check failed: {str(e)}")
            
        config_issues = []
        if not endpoint:
            config_issues.append("❌ Azure OpenAI Endpoint is missing")
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            config_issues.append("❌ Azure OpenAI API Key is missing")
        if not deployment_name:
            config_issues.append("❌ Deployment Name is missing")
        if use_private_endpoint and not private_ip:
            config_issues.append("❌ Private Endpoint IP is required when private endpoint is enabled")
            
        if config_issues:
            for issue in config_issues:
                st.error(issue)
        
        if st.button("🔍 Test Connection"):
            with st.spinner("Testing Azure OpenAI connection..."):
                try:
                    test_client = AzureOpenAIClient()
                    conn_info = test_client.get_connection_info()
                    
                    # Show connection details
                    st.info(f"🔗 Testing connection to: {conn_info['effective_endpoint']}")
                    if conn_info['use_private_endpoint']:
                        st.info(f"🔒 Using private endpoint via IP: {conn_info['private_endpoint_ip']}")
                    
                    success, message = test_client.test_connection()
                    
                    if success:
                        st.success("✅ Connection successful! Azure OpenAI is working perfectly.")
                        st.balloons()
                    else:
                        st.error(f"❌ Connection failed: {message}")
                        
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
    
    # Network Information
    st.subheader("Network Information")
    with st.expander("Current Network Details"):
        try:
            import requests
            current_ip = requests.get('https://ifconfig.me', timeout=5).text.strip()
            st.info(f"**Current IP Address**: {current_ip}")
            st.warning("⚠️ This IP is dynamic and changes when Replit restarts")
            
            st.markdown(f"""
            **To whitelist this IP in Azure OpenAI:**
            1. Go to Azure Portal → Your OpenAI Resource → Networking
            2. Select "Selected networks"
            3. Add IP address: `{current_ip}`
            4. Click Save and wait 2-3 minutes
            
            **Note**: You'll need to update this when the IP changes.
            """)
        except Exception as e:
            st.error(f"Could not detect current IP: {str(e)}")
    
    # Database settings
    st.subheader("Database")
    with st.expander("Database Information"):
        st.info("Using SQLite database for local storage")
        
        # Database stats
        analyses_count = len(db_manager.get_all_analyses())
        st.metric("Total Analyses", analyses_count)
        
        if st.button("Clear All Data", type="secondary"):
            if st.warning("This will delete all analysis data. Are you sure?"):
                db_manager.clear_all_data()
                st.success("All data cleared!")
                st.rerun()
    
    # Evaluation Criteria
    st.subheader("Evaluation Criteria")
    criteria = eval_engine.get_criteria_descriptions()
    
    for i, (name, description) in enumerate(criteria.items(), 1):
        with st.expander(f"Criterion {i}: {name}"):
            st.write(description)

if __name__ == "__main__":
    main()
