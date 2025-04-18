# App.py

import pandas as pd
import google.generativeai as genai
import streamlit as st
import io
import re
# from fpdf import FPDF
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from io import BytesIO
import matplotlib.pyplot as plt

# --- Configure Gemini ---
# genai.configure(api_key="AIzaSyD_MzbcVf7HlCHzJRQJoByKAgf9w9x-HBo")
genai.configure(api_key="AIzaSyCBjUZDQvG6uezrZ9sk0cnYKL68o-1KqQY")  #HCP
model = genai.GenerativeModel("gemini-2.0-flash",
    system_instruction="""
    You are an AI assistant helping with business data analysis.
    Avoid unnecessary data description. Always give numeric insights,
    prioritize visual understanding, and provide top/bottom performer tables and strategic recommendations.
    """)

# --- Streamlit UI ---
st.title("📊 Multi-Agent Excel AI Insights Dashboard")
st.write("Upload an Excel file and describe what you need. The AI system will assign the right agents automatically.")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])
user_prompt = st.text_area("💬 Your analysis prompt", "Analyze the dataset and provide insights and visualizations.")

# --- Main Coordinator Agent ---
def main_agent(user_prompt, df):
    planning_prompt = f"""
    You are a coordinator AI agent. The user prompt is:

    "{user_prompt}"

    Dataset preview:
    {df.head(5).to_markdown(index=False)}

    Decide what each of the following sub-agents should do:
    1. Insight Agent (Find some valuable insight and recommendations. Do not show data overview. Give numbers for each insights. Also provide the top and bottom performers.
Generate a sales report with data in tables )
    2. Chart Agent
    3. Report Agent (Generate a sales report with data in tables)
    4. Download Agent

    Reply in this format:
    - insight_task: [text or "none"]
    - chart_task: [text or "none"]
    - report_task: [text or "none"]
    - download_task: [text or "none"]
    """
    response = model.generate_content(planning_prompt)
    return response.text

# --- Parse Plan from Main Agent ---
def parse_plan(plan_text):
    tasks = {}
    for key in ["insight_task", "chart_task", "report_task", "download_task"]:
        match = re.search(rf"{key}:\s*(.*)", plan_text)
        tasks[key] = match.group(1).strip() if match else "none"
    return tasks

# --- Sub-Agents ---
def insight_agent(task, df):
    prompt = f"""
You are a data analysis expert AI. Your goal is to extract valuable **insights and recommendations** from the provided dataset. Follow these guidelines:

1. **Do NOT show data overview.**
2. **Always give numerical insights.** For example: "Product A has the highest sales: ₹1.2M" or "Sales dropped by 25% in Q2".
3. **Provide top 5 and bottom 5 performers** in a clean **markdown table** format if possible.
4. **Give actionable recommendations** based on the insights. Use bullets.
5. Ensure the content is direct, structured, and meaningful for business decisions.

6. **Trends & Patterns**: 
   - **Time-Based Insights**: Mention any noticeable trends related to specific periods (e.g., monthly, quarterly, yearly).
   - **Anomalies**: Point out any anomalies or outliers that may require attention.
   - Example: "Sales in Region X saw an unexpected spike of 50% during the summer months."

7. **Comparative Analysis**: 
   - Compare performance across categories or time periods. Example: "Sales in Q2 were 10% higher than Q1, primarily due to the introduction of Product Y."
   
8. **Correlation Insights**: Identify any correlations that might exist between variables (e.g., Sales and Marketing Spend). Example: "There is a strong positive correlation (r = 0.85) between advertising spend and sales for Product A."

9. **Segmentation Insights**: 
   - Segment the data based on available categories (e.g., regions, demographics, or other relevant segmentation).
   - Example: "Region Z saw the highest growth, while Region X underperformed in Q3."

10. **Opportunity Identification**: 
   - Identify areas that show potential for growth or improvement based on trends or data.
   - Example: "Despite being one of the bottom performers, Product Z has shown a steady increase in customer interest, signaling an opportunity for product optimization."

**Ensure that the insights provided are actionable, clear, and help in making business decisions.**

---

User Task:
{task}

---

Dataset Preview (First 20 Rows):
{df.head(20).to_markdown(index=False)}
"""
    return model.generate_content(prompt).text


def chart_agent(task, df):
    prompt = f"""
You are a chart suggestion AI. Based on the given data and task, recommend the most insightful charts.
Follow this format:
- Chart Title: ...
- Type: [Bar, Line, Pie, Scatter, Histogram]
- X-Axis: ...
- Y-Axis: ...
- Insight: What this chart will show

Suggest **3 to 5 charts** based on fields.

---

User Task:
{task}

Dataset (first 20 rows):
{df.head(20).to_markdown(index=False)}
"""
    return model.generate_content(prompt).text



def render_chart_from_task(task_text, df):
    task_text = task_text.lower()

    rendered = False

    if "bar chart" in task_text or "bar graph" in task_text:
        st.subheader("🔹 Bar Chart")
        for col in df.select_dtypes(include='object').columns:
            for num_col in df.select_dtypes(include='number').columns:
                chart_data = df.groupby(col)[num_col].sum().sort_values(ascending=False)
                st.write(f"Bar Chart of {num_col} by {col}")
                st.bar_chart(chart_data)
                rendered = True
                break
            if rendered:
                break

    if "line chart" in task_text or "trend" in task_text:
        st.subheader("🔹 Line Chart")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            for num_col in df.select_dtypes(include='number').columns:
                trend_data = df.groupby("date")[num_col].sum().reset_index()
                st.line_chart(trend_data.set_index("date"))
                rendered = True
                break

    if "pie chart" in task_text:
        st.subheader("🔹 Pie Chart")
        for col in df.select_dtypes(include='object').columns:
            pie_data = df[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            rendered = True
            break

    if "histogram" in task_text:
        st.subheader("🔹 Histogram")
        for num_col in df.select_dtypes(include='number').columns:
            fig, ax = plt.subplots()
            ax.hist(df[num_col].dropna(), bins=20, edgecolor='black')
            ax.set_title(f"Histogram of {num_col}")
            st.pyplot(fig)
            rendered = True
            break

    if "scatter" in task_text or "relationship" in task_text:
        st.subheader("🔹 Scatter Plot")
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.pyplot(fig)
            rendered = True

    if not rendered:
        st.warning("⚠️ No valid chart pattern detected in agent response.")

class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        self.set_font("DejaVu", "", 12)

def download_agent_pdf(user_prompt, plan_text, insights, chart_response, report):
    pdf = UnicodePDF()
    pdf.add_page()

    def write_section(title, content):
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=title, ln=True)
    
        pdf.set_font("Arial", size=12)
        cleaned_content = clean_text(content)
    
        # Split into manageable chunks
        lines = cleaned_content.split("\n")
        for line in lines:
            # Manually wrap long lines (max length 150 characters)
            if len(line) > 150:
                wrapped = re.findall('.{1,150}', line)  # Split line every 150 characters
                for chunk in wrapped:
                    pdf.multi_cell(0, 10, chunk)
            else:
                pdf.multi_cell(0, 10, line)
    
        pdf.ln(5)
    


    write_section("User Prompt", clean_text(user_prompt))
    write_section("Plan", clean_text(plan_text))
    write_section("Insights", clean_text(insights))
    write_section("Chart Analysis", clean_text(chart_response + "generate useful charts for sales analysis"))
    write_section("Report", clean_text(report))


    # Output PDF to BytesIO
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output


def report_agent(task, insights):
    prompt = f"{task}\n\nUse the insights below to generate a comprehensive report:\n{insights}"
    return model.generate_content(prompt).text

def download_agent(df):
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def download_agent_pdf(user_prompt, plan_text, insights, chart_response, report):
    pdf = FPDF()
    pdf.add_page()

    def write_section(title, content):
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=title, ln=True)
        pdf.set_font("Arial", size=12)
        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.ln(5)

    write_section("User Prompt", user_prompt)
    write_section("Plan", plan_text)
    write_section("Insights", insights)
    write_section("Chart Analysis", chart_response + "\n\n📊 Generated useful charts for sales analysis")
    write_section("Report", report)

    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    return pdf_output

def clean_text(text):
    # Remove emojis and unwanted symbols (except basic punctuation and common letters/numbers)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\u20B9"                 # ₹ symbol
        "\u20AC"                 # € symbol
        "\u00A3"                 # £ symbol
        "\u00A5"                 # ¥ symbol
        "\u200b"                 # zero-width space
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


# --- Main App Flow ---
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower()

    st.write("### 🧾 Data Preview")
    st.dataframe(df)

    plan_text = main_agent(user_prompt, df)
    tasks = parse_plan(plan_text)

    st.write("### 🧠 Task Planning by Main Agent")
    st.code(plan_text)

    # Insights
    if tasks["insight_task"].lower() != "none":
        insights = insight_agent(tasks["insight_task"], df)
        st.write("### 🔍 Insights")
        st.write(insights)
    else:
        insights = ""

    # Charts
    if tasks["chart_task"].lower() != "none":
        chart_response = chart_agent(tasks["chart_task"], df)
        st.write("### 📈 Chart Agent Suggestions")
        st.write(chart_response)

        # Optional: Show default 3-5 charts based on column types
        st.write("### 📊 Auto Visualizations Based on Agent Suggestion")
        render_chart_from_task(chart_response, df)

        # st.write("### 📊 Sample Visualizations")
        if "category" in df.columns and "sales" in df.columns:
            st.subheader("1. Sales by Category")
            st.bar_chart(df.groupby("category")["sales"].sum())

        if "date" in df.columns and "sales" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            st.subheader("2. Sales Over Time")
            st.line_chart(df.groupby("date")["sales"].sum().reset_index().set_index("date"))

        if "category" in df.columns:
            st.subheader("3. Category Distribution")
            st.pyplot(df["category"].value_counts().plot.pie(autopct='%1.1f%%').figure)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("4. 📊 Histogram Column", numeric_cols)
            st.subheader(f"4. Histogram of {selected_col}")
            st.pyplot(df[selected_col].plot.hist(bins=20, edgecolor='black').figure)

        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("5. X-axis for Scatter", numeric_cols, index=0)
            y_axis = st.selectbox("5. Y-axis for Scatter", numeric_cols, index=1)
            st.subheader(f"5. Scatter: {x_axis} vs {y_axis}")
            st.pyplot(df.plot.scatter(x=x_axis, y=y_axis, alpha=0.6).figure)

    # Report
    if tasks["report_task"].lower() != "none":
        report = report_agent(tasks["report_task"], insights)
        st.write("### 📝 Report")
        st.write(report)

    # Download
    if tasks["download_task"].lower() != "none":
        # output = download_agent(df)
        # st.download_button(label="📥 Download Analyzed File", data=output, file_name="Analyzed_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf_file = download_agent_pdf(user_prompt, plan_text, insights, chart_response, report)
        st.download_button(
    label="📄 Download PDF",
    data=pdf_file,
    file_name="agent_report.pdf",
    mime="application/pdf")
