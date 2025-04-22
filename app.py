import os
import re
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from io import BytesIO
import textwrap
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

# Gemini AI Config
genai.configure(api_key=os.getenv("GEMINI-KEY"))
model = genai.GenerativeModel("gemini-2.0-flash",
    system_instruction="""
    You are an AI assistant helping with business data analysis.
    Avoid unnecessary data description. Always give numeric insights,
    prioritize visual understanding, and provide top/bottom performer tables and strategic recommendations.
    """)

# Streamlit UI
st.title("📊 Multi-Agent Excel AI Insights Dashboard")
st.write("Upload an Excel file and describe what you need. The AI system will assign the right agents automatically.")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])
user_prompt = st.text_area("💬 Your analysis prompt", "Analyze the dataset and provide insights and visualizations.")

# Main Coordinator Agent
def main_agent(user_prompt, df):
    planning_prompt = f"""
    You are a coordinator AI agent. The user prompt is:

    "{user_prompt}"

    Dataset preview:
    {df.head(5).to_markdown(index=False)}

    Decide what each of the following sub-agents should do:
    1. Insight Agent
    2. Chart Agent
    3. Report Agent
    4. Download Agent

    Reply in this format:
    - insight_task: [text or "none"]
    - chart_task: [text or "none"]
    - report_task: [text or "none"]
    - download_task: [text or "none"]
    """
    response = model.generate_content(planning_prompt)
    return response.text

def parse_plan(plan_text):
    tasks = {}
    for key in ["insight_task", "chart_task", "report_task", "download_task"]:
        match = re.search(rf"{key}:\s*(.*)", plan_text)
        tasks[key] = match.group(1).strip() if match else "none"
    return tasks

# Sub-Agents
def insight_agent(task, df):
    prompt = f"""
    You are a data analysis expert AI. Extract valuable insights and recommendations from the dataset. Include:
    - Numeric insights
    - Top/Bottom performers in markdown tables
    - Time trends, anomalies
    - Correlation or segment performance
    - Actionable recommendations

    User Task:
    {task}

    Dataset Preview:
    {df.head(20).to_markdown(index=False)}
    """
    return model.generate_content(prompt).text

def chart_agent(task, df):
    prompt = f"""
    You are a chart suggestion AI. Suggest 3 to 5 insightful charts:
    - Chart Title
    - Type: [Bar, Line, Pie, Scatter, Histogram]
    - X-Axis / Y-Axis
    - Insight it reveals

    Task:
    {task}

    Dataset:
    {df.head(20).to_markdown(index=False)}
    """
    return model.generate_content(prompt).text

def report_agent(task, insights):
    prompt = f"{task}\n\nUse the insights below to generate a business report:\n{insights}"
    return model.generate_content(prompt).text

def render_chart_from_task(task_text, df):
    task_text = task_text.lower()
    rendered = False

    if "bar" in task_text:
        st.subheader("🔹 Bar Chart")
        for col in df.select_dtypes(include='object'):
            for num_col in df.select_dtypes(include='number'):
                st.bar_chart(df.groupby(col)[num_col].sum().sort_values(ascending=False))
                rendered = True
                break
            if rendered: break

    if "line" in task_text or "trend" in task_text:
        st.subheader("🔹 Line Chart")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            for num_col in df.select_dtypes(include='number'):
                st.line_chart(df.groupby("date")[num_col].sum().reset_index().set_index("date"))
                rendered = True
                break

    if "pie" in task_text:
        st.subheader("🔹 Pie Chart")
        for col in df.select_dtypes(include='object'):
            pie_data = df[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
            rendered = True
            break

    if "histogram" in task_text:
        st.subheader("🔹 Histogram")
        for col in df.select_dtypes(include='number'):
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=20, edgecolor='black')
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
            rendered = True
            break

    if "scatter" in task_text or "relationship" in task_text:
        st.subheader("🔹 Scatter Plot")
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6)
            ax.set_xlabel(num_cols[0])
            ax.set_ylabel(num_cols[1])
            ax.set_title(f"{num_cols[0]} vs {num_cols[1]}")
            st.pyplot(fig)
            rendered = True

    if not rendered:
        st.warning("⚠️ No valid chart pattern detected in agent response.")

# Unicode PDF
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Helvetica", "", 11)

    def write_section(self, title, content):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 8, content, align="L")

def clean_text(text):
    emoji_pattern = re.compile(r'[^\x00-\x7F]+')
    return emoji_pattern.sub('', text)

def download_agent_pdf(user_prompt, plan_text, insights, chart_response, report):
    pdf = UnicodePDF()
    pdf.write_section("User Prompt", clean_text(user_prompt))
    pdf.write_section("Task Planning", clean_text(plan_text))
    pdf.write_section("Insights", clean_text(insights))
    pdf.write_section("Chart Agent Suggestions", clean_text(chart_response))
    pdf.write_section("Report", clean_text(report))

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Excel Download
def download_agent(df):
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

# --- Main App Execution ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower()

    st.write("### 🧾 Data Preview")
    st.dataframe(df)

    plan_text = main_agent(user_prompt, df)
    tasks = parse_plan(plan_text)

    st.write("### 🧠 Task Planning")
    st.code(plan_text)

    insights = chart_response = report = ""

    if tasks["insight_task"].lower() != "none":
        insights = insight_agent(tasks["insight_task"], df)
        st.write("### 🔍 Insights")
        st.write(insights)

    if tasks["chart_task"].lower() != "none":
        chart_response = chart_agent(tasks["chart_task"], df)
        st.write("### 📈 Chart Agent Suggestions")
        st.write(chart_response)
        st.write("### 📊 Visualizations")
        render_chart_from_task(chart_response, df)

    if tasks["report_task"].lower() != "none":
        report = report_agent(tasks["report_task"], insights)
        st.write("### 🧾 Report")
        st.write(report)

    st.write("---")
    st.subheader("📥 Download Options")

    st.download_button("Download Excel", download_agent(df), "analysis_data.xlsx")

    if st.button("📄 Generate PDF Report"):
        pdf_file = download_agent_pdf(user_prompt, plan_text, insights, chart_response, report)
        st.download_button("Download PDF Report", pdf_file, "ai_analysis_report.pdf", mime="application/pdf")
