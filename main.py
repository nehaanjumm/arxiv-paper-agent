import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from src.graph import build_graph  # noqa: E402

if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]) or input("Enter a topic or arXiv ID/URL: ")
    app = build_graph()
    app.invoke({"user_input": user_input, "chat_history": []},
               config={"recursion_limit": 100})