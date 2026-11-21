"""
Keyword search vs meaning (embedding) search on 50 Zoho documents.

Run:
    python search_compare.py

Keyword search = BM25 (classic word matching, no AI).
Meaning search = sentence embeddings (all-MiniLM-L6-v2) + cosine similarity.
Results are printed and also saved to results.md for your write-up.
"""
import math
import re
from collections import Counter

import numpy as np

DOCS = [
    'Zoho Analytics - Q1 2026 Product Update',
    'Introducing Leaderboard in Zoho Publish',
    'Admin reports in Zoho Mail: From user accounts to team communication',
    'App Spotlight: 360 SMS for Zoho CRM',
    'Reduce SLA breaches with Zoho Desk and Zoho Contracts integration',
    'Announcing SAP Business One integrations in Zoho Flow',
    'Presenting Zoho Sprints 3.0: Enterprise-ready Agile Project Management',
    'App Spotlight: Clearout Email Verifier for Zoho CRM',
    'App Spotlight: Tap Payments for Zoho Books',
    'Zoho Analytics meets Zoho ERP: Get actionable insights from your Zoho ERP data',
    'No more silos: How Zoho Workplace creates a unified workflow',
    'Rewind with us - Zoho Analytics in 2025',
    'Recap 2025: Zoho Assist',
    'Build AI-powered workflows with Zoho Cliq MCP and the Cliq developer platform',
    'Introducing Meeting Co-Host support in Zoho Calendar',
    'Zoho DataPrep: A look back at a transformative 2025 and a glimpse into 2026',
    'Event-driven document automation using webhooks in Zoho Writer',
    'Introducing Accessibility Controls in Zoho Vault',
    'AI that actually helps: Practical intelligence in Zoho Workplace',
    'From inventory reports to inventory intelligence with Zoho Inventory and Zoho Analytics',
    'The wait is over: Zoho Tables is now available in Zoho One',
    'AI that moves work forward: Inside Zoho Cliq 7.0',
    'Meet the new Zoho Expense: Integrated business travel with native AI across every layer',
    'Introducing Zoho PageSense MCP',
    'Introducing Zoho PDF Editor Desktop App for Windows, Mac, and Linux',
    'Frequently asked questions on ABM for Zoho CRM',
    'Frequently Asked Questions on CRM for Everyone',
    'FAQs: CRM Onboarding',
    'FAQs: Zoho CRM Integration (Bookings, Projects, Desk, Survey, Google Ads)',
    'FAQs: Zoho CRM Customization (Kiosk Studio, Modules, Page layouts)',
    'Why am I not able to send invoices, quotes, etc. when I use an AOL email address as my primary email address?',
    'How do I change the base currency of my organization?',
    "How can I change my organization's fiscal year?",
    'How do I delete my Zoho Books organization?',
    'How do I enable Multi-Factor Authentication for my Zoho Books account?',
    'I forgot my password. How do I reset it?',
    'How do I reset my organization and start afresh?',
    'Do you store my credit card details?',
    'How do I prevent others from editing the PDF files sent from Zoho Books?',
    "How do I record my employees' salaries in Zoho Books?",
    "How do I make my company's logo visible on all my transactions?",
    'How do I remove the Zoho branding from the emails and PDFs of the transactions?',
    "I'm trying to send invoices from Zoho Books, but they had bounced back due to a bad mailbox. What should I do?",
    'Is it possible to change the email address with which I access my Zoho Books account?',
    "My recurring invoices aren't being sent automatically, and it says there are no contact persons associated. Is it mandatory to provide contact person details?",
    'Do you have Payroll in Zoho Books?',
    'What is help center in Desk?',
    'Can end users view tickets in the Help Center?',
    'Spam mails come into my Inbox. How can I stop them?',
    'Is Zoho Books WCAG compliant?',
]

QUERIES = [
    # Meaning wins: almost no shared words with the right document.
    "stop using Zoho Books and remove my company data",
    "smart assistants that automate my work",
    "unwanted junk messages keep arriving",
    # Bonus: an exact rare term. Keyword search is expected to do well here.
    "WCAG",
]
TOP_K = 3

STOPWORDS = set("""a an and are as at be by do does for from how i in is it my of on or that the
to up we what when where which who why will with you your me can this these those so if not no""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


class BM25:
    """Plain BM25: scores a document higher when it contains the query's exact words."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for w in tokenize(query):
            if w not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg)
                    out[i] += self.idf[w] * f * (self.k1 + 1) / norm
        return out


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def short(t, n=80):
    return t if len(t) <= n else t[: n - 1] + "…"


def top(scores, k):
    idx = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
    return idx


def main():
    texts = DOCS
    bm25 = BM25(texts)
    model = get_model()
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True))

    md = ["# Keyword vs meaning search (50 Zoho documents)\n"]
    for q in QUERIES:
        kw = bm25.scores(q)
        em = vectors @ np.asarray(model.encode([q], normalize_embeddings=True))[0]
        kw_top, em_top = top(kw, TOP_K), top(em, TOP_K)

        print(f'\nQUERY: "{q}"')
        print("  Keyword (BM25):")
        if not kw_top:
            print("    (no results: none of the query words appear in any document)")
        for i in kw_top:
            print(f"    {kw[i]:5.2f}  [{i + 1}] {short(texts[i])}")
        print("  Meaning (embeddings):")
        for i in em_top:
            print(f"    {em[i]:5.2f}  [{i + 1}] {short(texts[i])}")
        same = bool(kw_top) and kw_top[0] == em_top[0]
        print("  -> Same #1 result" if same else "  -> Different #1 result")

        md.append(f'\n## Query: "{q}"\n')
        md.append("| Rank | Keyword (BM25) | Meaning (embeddings) |\n|---|---|---|")
        for r in range(TOP_K):
            k_cell = f"{short(texts[kw_top[r]], 60)} ({kw[kw_top[r]]:.2f})" if r < len(kw_top) else "-"
            e_cell = f"{short(texts[em_top[r]], 60)} ({em[em_top[r]]:.2f})" if r < len(em_top) else "-"
            md.append(f"| {r + 1} | {k_cell} | {e_cell} |")
        md.append("\nWhy they differ: (write your explanation here)\n")

    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\nSaved results.md")


if __name__ == "__main__":
    main()