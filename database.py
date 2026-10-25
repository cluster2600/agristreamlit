import sqlite3
import os
import pdfplumber



def test_pdf_data_retrieval(query):
    """Test retrieving data from PDFs stored in the database."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT question, response FROM farming_info WHERE question LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()

    conn.close()

    if results:
        for question, response in results:
            print(f"\n🔹 Question: {question}\n💡 Answer: {response[:500]}...")  # Limit output for readability
    else:
        print("⚠️ No relevant information found. Try rephrasing your query.")

# Initialize Database
def init_db():
    """Create the farming_info table if it doesn't exist."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS farming_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            response TEXT
        )
    ''')

    conn.commit()
    conn.close()


# Insert Predefined Farming Data
def insert_farming_data():
    """Insert predefined farming questions and answers into the database."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    farming_data = [
        # 🌽 Maize-related questions
        ("How can I grow maize?", "To grow maize, ensure well-drained soil, plant during the rainy season, and use quality seeds."),
        ("Best time to plant maize?", "The best time to plant maize is at the onset of the rainy season."),
        ("How to control pests in maize?", "Use crop rotation, certified seeds, and organic pesticides."),
        ("What soil is best for maize?", "Loamy, well-drained soil with a pH of 5.5 to 7.0 is best for maize farming."),

        # 🥔 Potato-related questions
        ("How can I grow potatoes?", "Potatoes grow best in loose, well-drained soil. Plant tubers in rows with full sunlight."),
        ("Best time to plant potatoes?", "Plant potatoes in early spring when soil temperature is above 45°F (7°C)."),
        ("How to control pests in potatoes?", "Use certified seeds, crop rotation, and organic insecticides."),
        ("What soil is best for potatoes?", "Sandy loam soil with a pH between 5.0 and 6.5 is ideal for potatoes."),

        # 🥕 Carrot-related questions
        ("How can I grow carrots?", "Carrots need loose, sandy soil. Sow seeds directly at a depth of 1/4 inch."),
        ("Best time to plant carrots?", "Plant carrots in early spring or late summer."),
        ("How to control pests in carrots?", "Use floating row covers and companion planting."),
        ("What soil is best for carrots?", "Well-drained, sandy loam soil with a pH between 6.0 and 6.8.")
    ]

    for question, response in farming_data:
        cursor.execute("INSERT OR IGNORE INTO farming_info (question, response) VALUES (?, ?)", (question, response))

    conn.commit()
    conn.close()


# Extract Text from PDFs
def extract_text_from_pdfs():
    """Extracts text from PDFs in the 'kalro_pdfs' folder."""
    data = {}
    pdf_folder = "kalro_pdfs"

    if not os.path.exists(pdf_folder):
        print("⚠️ PDF folder does not exist!")
        return {}

    for pdf_file in os.listdir(pdf_folder):
        pdf_path = os.path.join(pdf_folder, pdf_file)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join([page.extract_text() or "" for page in pdf.pages]).strip()
                if text:
                    data[pdf_file.replace(".pdf", "").strip()] = text  # Store filename as question
        except Exception as e:
            print(f"⚠️ Error reading {pdf_file}: {e}")

    return data


# Store Extracted PDF Content in Database
def store_in_db(data):
    """Store extracted PDF data into the database with meaningful question-based entries."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    for filename, text in data.items():
        # Extract the first 10 words as the "question" if no clear question exists
        question = " ".join(text.split()[:10]) + "?"  # Convert first sentence to a question-like format
        response = text.strip()

        if question and response:
            cursor.execute("INSERT OR IGNORE INTO farming_info (question, response) VALUES (?, ?)", (question, response))
            print(f"✅ Stored: {question}")  # Debugging print

    conn.commit()
    conn.close()



def search_farming_info(query):
    """Search the database for a query, handling multiple keywords and partial matches."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    # Split query into individual keywords for better matching
    keywords = query.split()
    conditions = " OR ".join(["question LIKE ? OR response LIKE ?" for _ in keywords])
    params = sum([('%' + word + '%', '%' + word + '%') for word in keywords], ())

    cursor.execute(f"""
        SELECT response FROM farming_info
        WHERE {conditions}
    """, params)

    results = cursor.fetchall()
    conn.close()

    if results:
        responses = "\n\n".join([result[0][:500] + "..." for result in results])  # Limit long responses
        return f"💡 Found {len(results)} match(es):\n{responses}"
    else:
        return f"⚠️ No relevant farming info found for '{query}'. Try using general terms like 'wilt' or 'potato disease'."

def get_farming_info(query):
    """Search database first, fallback to AI if no match."""
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()

    query = query.lower()
    cursor.execute("SELECT response FROM farming_info WHERE question LIKE ?", ('%' + query + '%',))
    data = cursor.fetchone()

    conn.close()

    if data:
        return data[0]  # Return database answer if found
    else:
        return get_chatbot_response(query)  # Use AI if no match


# Initialize & Populate Database
if __name__ == "__main__":
    init_db()  # Ensure the table exists
    insert_farming_data()  # Insert predefined data

    # Extract PDF data and store it
    extracted_data = extract_text_from_pdfs()
    if extracted_data:
        store_in_db(extracted_data)

    # Test a search query
    user_query = "How can I grow maize?"
    print(f"\n🔍 Query: {user_query}\n💡 Answer: {search_farming_info(user_query)}")
