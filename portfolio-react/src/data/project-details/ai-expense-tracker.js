export default {
  modules: [
    {
      name: 'Authentication',
      description: 'Handles registration, login, token-based access, and user-specific financial data.',
    },
    {
      name: 'Expense Management',
      description: 'Creates, updates, categorizes, and stores individual income and expense transactions.',
    },
    {
      name: 'NLP Expense Parser',
      description: 'Converts natural-language entries such as "spent 500 on groceries" into structured transaction data.',
    },
    {
      name: 'Financial Analytics',
      description: 'Aggregates transactions into category breakdowns, spending trends, and planning insights.',
    },
  ],
  technologies: [
    { name: 'React', purpose: 'Builds the responsive client interface and analytics dashboard.' },
    { name: 'Django REST Framework', purpose: 'Provides authenticated REST endpoints for financial data.' },
    { name: 'SQLite', purpose: 'Persists user accounts, transaction records, and categories.' },
    { name: 'Python', purpose: 'Powers the backend logic and natural-language processing workflow.' },
    { name: 'NLP', purpose: 'Extracts expense details from free-form user input.' },
  ],
  functionality: [
    'Securely create an account and access only personal financial records.',
    'Add expenses manually or enter them in natural language for automatic parsing.',
    'Organize transactions by category and review them through REST-backed views.',
    'Monitor spending patterns with interactive category and trend analytics.',
    'Use financial summaries to support day-to-day budgeting and planning.',
  ],
};
