from pydantic import BaseModel

class CreditScoreInputData(BaseModel):
    annual_income: float
    monthly_inhand_salary: float
    num_bank_accounts: int
    num_credit_card: int
    interest_rate: int
    num_of_loan: int 
    delay_from_due_date: int
    num_of_delayed_payment: int
    changed_credit_limit: float
    num_credit_inquiries: int
    outstanding_debt: float
    credit_utilization_ratio: float
    total_emi_per_month: float
    amount_invested_monthly: float
    monthly_balance: float

class CreditScoreOutputData(BaseModel):
    score: str

class InsuranceInputData(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class InsuranceOutputData(BaseModel):
    charges: float