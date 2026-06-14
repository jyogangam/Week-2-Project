# Sample Enterprise Policy Corpus

This folder contains a fictional policy corpus for building and evaluating an
Enterprise Policy Q&A bot. The documents are written to support several retrieval
patterns:

| Document | Main topics | Useful retrieval behavior |
|---|---|---|
| HR-001 Employee Handbook | PTO, hours, policy routing | Broad questions and cross-references |
| HR-014 Remote Work Policy | eligibility, locations, equipment | Detailed rules and exceptions |
| FIN-008 Travel and Expense Policy | travel, receipts, meals | Numeric limits and policy precedence |
| SEC-003 Information Security Policy | classification, AI, incidents | Urgent procedures and restrictions |
| LEG-011 Data Retention and Privacy Policy | retention, legal holds, privacy | Multi-step and cross-policy answers |
| OPS-005 Onboarding Guide | new-hire tasks and timing | Similar wording with important boundaries |

All organizations, policies, people, and rules in this corpus are fictional.

## Built-In Evaluation Opportunities

- **Single-document lookup:** "How many PTO days can carry over?"
- **Cross-document answer:** "What should I do if I lose my laptop while
  traveling?"
- **Policy precedence:** "Can my manager approve reimbursement for pet care?"
- **Ambiguous wording:** "Can a new employee work remotely?"
- **Multi-hop answer:** "How long are records from a security investigation kept,
  and what should I do before deleting them?"
- **Unanswerable question:** "What dental insurance plan does Northstar offer?"

The document headers provide useful metadata fields for filtering and citations:
document ID, owner, version, effective date, and audience.
