from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the nodes (variables) in our Bayesian Network
# representing factors that could influence cyber risk
model = DiscreteBayesianNetwork([
        ('Vulnerability', 'Exploit'),
        ('ThreatActor', 'Exploit'),
        ('Exploit', 'DataBreach'),
        ('SecurityControls', 'DataBreach')
    ]
)

# Define the Conditional Probability Distributions (CPDs) for each node

# CPD for Vulnerability (High/Low)
cpd_vulnerability = TabularCPD(
    variable='Vulnerability',
    variable_card=2,
    values=[
        [0.6], # P(Vulnerability=High)
        [0.4], # P(Vulnerability=Low)
    ],
    state_names={
        'Vulnerability': ['High', 'Low']
    },
)

# CPD for ThreatActor (Sophisticated/Opportunistic)
cpd_threatactor = TabularCPD(
    variable='ThreatActor',
    variable_card=2,
    values=[
        [0.7], # P(ThreatActor=Sophisticated)
        [0.3], # P(ThreatActor=Opportunistic)
    ],
    state_names={
        'ThreatActor': ['Sophisticated', 'Opportunistic']
    }
)

# CPD for Exploit (Successful/Failed) - depends on Vulnerability and ThreatActor
cpd_exploit = TabularCPD(
    variable='Exploit',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.3],  # P(Exploit=Successful | V=H, TA=S), P(E=S | V=H, TA=O), P(E=S | V=L, TA=S), P(E=S | V=L, TA=O)
        [0.1, 0.3, 0.4, 0.7],
    ],  # P(Exploit=Failed | V=H, TA=S), P(E=F | V=H, TA=O), P(E=F | V=L, TA=S), P(E=F | V=L, TA=O)
    evidence=['Vulnerability', 'ThreatActor'],
    evidence_card=[2, 2],
    state_names={
        'Exploit': ['Successful', 'Failed'],
        'Vulnerability': ['High', 'Low'],
        'ThreatActor': ['Sophisticated', 'Opportunistic']
    }
)

# CPD for SecurityControls (Effective/Ineffective)
cpd_securitycontrols = TabularCPD(
    variable='SecurityControls',
    variable_card=2,
    values=[
        [0.75], # P(SecurityControls=Effective)
        [0.25], # P(SecurityControls=Ineffective)
    ],
    state_names={
        'SecurityControls': ['Effective', 'Ineffective']
    }
)

# CPD for DataBreach (Yes/No) - depends on Exploit and SecurityControls
cpd_databreach = TabularCPD(
    variable='DataBreach',
    variable_card=2,
    values=[
        # P(DataBreach=Yes | E=S, SC=I), P(DB=Y | E=S, SC=E), P(DB=Y | E=F, SC=I), P(DB=Y | E=F, SC=E)
        [0.95, 0.65, 0.7, 0.3],
        # P(DataBreach=No | E=S, SC=I), P(DB=N | E=S, SC=E), P(DB=N | E=F, SC=I), P(DB=N | E=F, SC=E)
        [0.05, 0.35, 0.3, 0.7],
    ],
    evidence=['Exploit', 'SecurityControls'],
    evidence_card=[2, 2],
    state_names={
        'DataBreach': ['Yes', 'No'],
        'Exploit': ['Successful', 'Failed'],
        'SecurityControls': ['Effective', 'Ineffective']
    }
) # Note the order of states here matches the values

# Add the CPDs to the model
model.add_cpds(cpd_vulnerability, cpd_threatactor, cpd_exploit, cpd_securitycontrols, cpd_databreach)

# Check if the model is valid
print(f"Is the model valid? {model.check_model()}")

# Perform inference
infer = VariableElimination(model)

# What is the probability of a data breach occurring?
p_of_databreach = infer.query(
    variables=['DataBreach'],
)
print("\nProbability of a Data Breach:\n", p_of_databreach)

# What is the probability of a data breach given a high vulnerability?
p_of_databreach_given_high_vuln = infer.query(
    variables=['DataBreach'],
    evidence={'Vulnerability': 'High'},
)
print("\nProbability of a Data Breach given High Vulnerability:\n", p_of_databreach_given_high_vuln)

# What is the probability of a data breach given a successful exploit and ineffective security controls?
p_of_databreach_given_exploit_and_weak_controls = infer.query(
    variables=['DataBreach'],
    evidence={
        'Exploit': 'Successful',
        'SecurityControls': 'Ineffective',
    },
)
print("\nProbability of a Data Breach given Successful Exploit and Ineffective Security Controls:\n", p_of_databreach_given_exploit_and_weak_controls)

# What is the probability that the vulnerability was high given a data breach occurred?
p_of_high_vuln_given_databreach = infer.query(
    variables=['Vulnerability'],
    evidence={'DataBreach': 'Yes'},
)
print("\nProbability of High Vulnerability given a Data Breach:\n", p_of_high_vuln_given_databreach)
