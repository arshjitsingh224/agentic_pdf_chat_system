import streamlit as st
import os
import tempfile
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import json

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic PDF Chat System",
    page_icon="📄",
    layout="wide"
)

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-70b-8192"

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None

# State definition for LangGraph
class AgentState(TypedDict):
    question: str
    pdf_content: List[Document]
    retrieved_docs: List[str]
    web_search_results: str
    final_answer: str
    source_type: str
    reasoning: str

class PDFChatAgent:
    def __init__(self, groq_api_key: str, tavily_api_key: str):
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=groq_api_key,
            temperature=0.1
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.vector_store = None
        
        # Create the graph
        self.workflow = self._create_workflow()
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Process uploaded PDF and create vector store"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return splits
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def _create_workflow(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("pdf_search", self._search_pdf)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("llm_response", self._llm_response)
        workflow.add_node("generate_answer", self._generate_final_answer)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "pdf": "pdf_search",
                "web": "web_search", 
                "llm": "llm_response"
            }
        )
        
        # Add edges to final answer generation
        workflow.add_edge("pdf_search", "generate_answer")
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("llm_response", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Analyze the query to determine the best routing strategy"""
        question = state["question"]
        
        # Create routing prompt
        routing_prompt = PromptTemplate(
            template="""
            Analyze the following question and determine the best way to answer it:
            
            Question: {question}
            
            Consider:
            1. Is this a question that would likely be answered by content in a PDF document?
            2. Is this a question requiring current/recent information that would need web search?
            3. Is this a general knowledge question that can be answered with LLM knowledge?
            
            Provide your analysis and routing decision in the following format:
            ANALYSIS: [Your reasoning]
            ROUTE: [pdf/web/llm]
            
            Guidelines:
            - Choose 'pdf' if the question seems document-specific or asks about content that would typically be in uploaded documents
            - Choose 'web' if the question requires current information, recent events, or real-time data
            - Choose 'llm' if it's general knowledge that doesn't require specific documents or current information
            """,
            input_variables=["question"]
        )
        
        response = self.llm.invoke(routing_prompt.format(question=question))
        
        # Parse the response to extract routing decision
        lines = response.content.split('\n')
        analysis = ""
        route = "llm"  # default
        
        for line in lines:
            if line.startswith("ANALYSIS:"):
                analysis = line.replace("ANALYSIS:", "").strip()
            elif line.startswith("ROUTE:"):
                route = line.replace("ROUTE:", "").strip().lower()
        
        state["reasoning"] = analysis
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Make routing decision based on analysis"""
        question = state["question"]
        
        # Simple routing logic - can be enhanced
        if self.vector_store is None:
            if any(keyword in question.lower() for keyword in ["current", "recent", "today", "latest", "news"]):
                return "web"
            else:
                return "llm"
        
        # If we have PDF content, try to determine if question is PDF-related
        if any(keyword in question.lower() for keyword in ["document", "pdf", "file", "content", "this document"]):
            return "pdf"
        elif any(keyword in question.lower() for keyword in ["current", "recent", "today", "latest", "news", "2024", "2025"]):
            return "web"
        else:
            return "pdf"  # Default to PDF search if we have content
    
    def _search_pdf(self, state: AgentState) -> AgentState:
        """Search in PDF content using RAG"""
        if self.vector_store is None:
            state["retrieved_docs"] = []
            state["source_type"] = "error"
            return state
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(state["question"], k=3)
            state["retrieved_docs"] = [doc.page_content for doc in docs]
            state["source_type"] = "pdf"
            return state
        except Exception as e:
            state["retrieved_docs"] = []
            state["source_type"] = "error"
            return state
    
    def _web_search(self, state: AgentState) -> AgentState:
        """Perform web search using Tavily"""
        try:
            search_results = self.tavily_client.search(
                query=state["question"],
                max_results=3,
                search_depth="basic"
            )
            
            # Format search results
            formatted_results = ""
            for result in search_results.get('results', []):
                formatted_results += f"Title: {result.get('title', '')}\n"
                formatted_results += f"Content: {result.get('content', '')}\n"
                formatted_results += f"URL: {result.get('url', '')}\n\n"
            
            state["web_search_results"] = formatted_results
            state["source_type"] = "web"
            return state
        except Exception as e:
            state["web_search_results"] = f"Error performing web search: {str(e)}"
            state["source_type"] = "error"
            return state
    
    def _llm_response(self, state: AgentState) -> AgentState:
        """Generate response using LLM knowledge only"""
        state["source_type"] = "llm"
        return state
    
    def _generate_final_answer(self, state: AgentState) -> AgentState:
        """Generate the final answer based on the source type"""
        question = state["question"]
        source_type = state["source_type"]
        
        if source_type == "pdf":
            context = "\n\n".join(state["retrieved_docs"])
            prompt = PromptTemplate(
                template="""
                Based on the following context from the PDF document, answer the question:
                
                Context:
                {context}
                
                Question: {question}
                
                Please provide a comprehensive answer based on the PDF content. If the context doesn't fully answer the question, say so.
                """,
                input_variables=["context", "question"]
            )
            response = self.llm.invoke(prompt.format(context=context, question=question))
            
        elif source_type == "web":
            context = state["web_search_results"]
            prompt = PromptTemplate(
                template="""
                Based on the following web search results, answer the question:
                
                Search Results:
                {context}
                
                Question: {question}
                
                Please provide a comprehensive answer based on the search results.
                """,
                input_variables=["context", "question"]
            )
            response = self.llm.invoke(prompt.format(context=context, question=question))
            
        else:  # llm knowledge
            prompt = PromptTemplate(
                template="""
                Answer the following question using your knowledge:
                
                Question: {question}
                
                Please provide a comprehensive and accurate answer.
                """,
                input_variables=["question"]
            )
            response = self.llm.invoke(prompt.format(question=question))
        
        state["final_answer"] = response.content
        return state
    
    def chat(self, question: str) -> Dict[str, Any]:
        """Main chat function"""
        initial_state = AgentState(
            question=question,
            pdf_content=st.session_state.pdf_content or [],
            retrieved_docs=[],
            web_search_results="",
            final_answer="",
            source_type="",
            reasoning=""
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "answer": final_state["final_answer"],
            "source_type": final_state["source_type"],
            "reasoning": final_state.get("reasoning", "")
        }

def main():
    st.title("🤖 Agentic PDF Chat System")
    st.markdown("Upload a PDF and ask questions! The system intelligently routes queries between PDF content, web search, and LLM knowledge.")
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 Configuration")
        groq_api_key = st.text_input("Groq API Key", type="password")
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
        if groq_api_key and tavily_api_key:
            if 'agent' not in st.session_state:
                st.session_state.agent = PDFChatAgent(groq_api_key, tavily_api_key)
            st.success("✅ APIs configured!")
        else:
            st.warning("Please enter both API keys to continue")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📄 PDF Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and 'agent' in st.session_state:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    documents = st.session_state.agent.process_pdf(uploaded_file)
                    st.session_state.pdf_content = documents
                    st.session_state.vector_store = st.session_state.agent.vector_store
                    st.success(f"✅ PDF processed! {len(documents)} chunks created.")
        
        # Show PDF status
        if st.session_state.pdf_content:
            st.info(f"📄 PDF loaded: {len(st.session_state.pdf_content)} chunks")
    
    with col2:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for i, (question, response) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}..."):
                st.write("**Question:**", question)
                st.write("**Answer:**", response["answer"])
                
                # Source indicator
                source_type = response["source_type"]
                if source_type == "pdf":
                    st.info("📄 **Source:** PDF Document")
                elif source_type == "web":
                    st.info("🌐 **Source:** Web Search")
                elif source_type == "llm":
                    st.info("🧠 **Source:** LLM Knowledge")
                
                # if response.get("reasoning"):
                #     st.write("**Routing Reasoning:**", response["reasoning"])
        
        # Chat input
        if 'agent' in st.session_state:
            question = st.text_input("Ask a question:", key="question_input")
            
            if st.button("Send") and question:
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.chat(question)
                    st.session_state.chat_history.append((question, response))
                    st.rerun()
        else:
            st.warning("Please configure API keys first!")
    
    # Instructions
    with st.expander("📋 How to use"):
        st.markdown("""
        1. **Configure API Keys**: Enter your Groq and Tavily API keys in the sidebar
        2. **Upload PDF** (optional): Upload a PDF document to enable document-based Q&A
        3. **Ask Questions**: The system will intelligently route your questions to:
           - 📄 **PDF Content**: For document-specific questions
           - 🌐 **Web Search**: For current information and recent events
           - 🧠 **LLM Knowledge**: For general knowledge questions
        
        **Features:**
        - Intelligent query routing using LangGraph
        - RAG-powered PDF search
        - Web search for current information
        - Source attribution for transparency
        - Conversational memory
        """)

if __name__ == "__main__":
    main()