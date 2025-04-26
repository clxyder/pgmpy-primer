from pgmpy.models import DiscreteBayesianNetwork  # Updated class name
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def main():
    """
    from: https://www.linkedin.com/pulse/overview-pgmpy-python-library-cybersecurity-risk-analysis-tim-layton-rriac/
    """

    # Step 1: Define the structure of the Bayesian Network
    model = DiscreteBayesianNetwork([
        ('ExternalThreat', 'DataBreach'),
        ('SystemVulnerability', 'DataBreach'),
        ('SecurityMeasures', 'SystemVulnerability')
    ])

    # Step 2: Define the Conditional Probability Distributions (CPDs)
    cpd_external = TabularCPD(
        variable='ExternalThreat',
        variable_card=2,
        values=[[0.5], [0.5]], # 50% chance for each state
        state_names={
            'ExternalThreat': ['Internal', 'External']
        },
    )

    cpd_security = TabularCPD(
        variable='SecurityMeasures',
        variable_card=2,
        values=[[0.7], [0.3]], # 70% chance for strong, 30% for weak
        state_names={
            'SecurityMeasures': ['Weak', 'Strong']
        },
    )

    # SystemVulnerability depends on SecurityMeasures only, according to the network structure
    cpd_vulnerability = TabularCPD(
        variable='SystemVulnerability',
        variable_card=2,
        values=[
            # P(SystemVulnerability=Low | SecurityMeasures=Weak), P(SV=Low | SM=Strong)
            [0.1, 0.9],
            # P(SystemVulnerability=High | SecurityMeasures=Weak), P(SV=High | SM=Strong)
            [0.9, 0.1],
        ],
        evidence=['SecurityMeasures'],
        evidence_card=[2],
        state_names={
            'SecurityMeasures': ['Weak', 'Strong'],
            'SystemVulnerability': ['Low', 'High'],
        },
    )

    # DataBreach depends on ExternalThreat and SystemVulnerability
    cpd_breach = TabularCPD(
        variable='DataBreach', variable_card=2,
        values=[
            # P(DataBreach=Yes | ExternalThreat=Internal, SystemVulnerability=Low), P(DB=Yes | ET=Internal, SV=High), P(DB=Yes | ET=External, SV=Low), P(DB=Yes | ET=External, SV=High)
            [0.01, 0.1, 0.4, 0.9], # Probabilities for high risk
            # P(DataBreach=No | ExternalThreat=Internal, SystemVulnerability=Low), P(DB=No | ET=Internal, SV=High), P(DB=No | ET=External, SV=Low),
            [0.99, 0.9, 0.6, 0.1], # Probabilities for low risk
        ],
        evidence=['SystemVulnerability', 'ExternalThreat'],
        evidence_card=[2, 2],
        state_names={
            'DataBreach': ['Yes', 'No'],
            'SystemVulnerability': ['Low', 'High'],
            'ExternalThreat': ['Internal', 'External']
        }
    )

    # Step 3: Add the CPDs to the model
    model.add_cpds(cpd_external, cpd_security, cpd_vulnerability, cpd_breach)

    # Step 4: Validate the model to ensure it's correctly structured
    if model.check_model():
        print("Model is valid.")
    else:
        print("Model is invalid.")

    # Step 5: Perform inference on the model
    inference = VariableElimination(model)

    # Query the probability of a Data Breach given strong Security Measures
    prob_breach = inference.query(
        variables=['DataBreach'],
        evidence={
            #'SecurityMeasures': 1,
            'SecurityMeasures': "Strong"
        },
    )
    print(prob_breach)


if __name__ == "__main__":
    main()
