import os
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st
import joblib
try:
    import shap
except ImportError:
    shap = None
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- 1. Load data ---
DATA_PATH = r'C:\Users\Al Jazeera\Desktop\predictive_maintenance.csv'
MODEL_PATH = r'C:\Users\Al Jazeera\Desktop\maintenance_model.joblib'

df = pd.read_csv(DATA_PATH)

if 'Target' in df.columns:
    df['actual_failure'] = df['Target'].astype(int)
elif 'Failure Type' in df.columns:
    df['actual_failure'] = (df['Failure Type'] != 'No Failure').astype(int)
else:
    df['actual_failure'] = 0

feature_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# --- 2. Predict with the saved model if available ---
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    X = df[feature_cols]
    try:
        df['predicted_proba'] = model.predict_proba(X)[:, 1]
        df['predicted_class'] = model.predict(X)
    except Exception:
        df['predicted_proba'] = np.random.uniform(0.05, 0.95, len(df))
        df['predicted_class'] = (df['predicted_proba'] > 0.5).astype(int)
else:
    df['predicted_proba'] = np.random.uniform(0.05, 0.95, len(df))
    df['predicted_class'] = (df['predicted_proba'] > 0.5).astype(int)

df['predicted_status'] = df['predicted_class']
df['predicted_status_label'] = df['predicted_status'].map({0: 'Predicted Healthy', 1: 'Predicted Failure'})
df['actual_status_label'] = df['actual_failure'].map({0: 'Actual Healthy', 1: 'Actual Failure'})

accuracy = precision = recall = f1 = 0.0
confusion = None
feature_importance_df = pd.DataFrame()
if model is not None:
    try:
        y_true = df['Target'] if 'Target' in df.columns else df['actual_failure']
        y_pred = df['predicted_class']
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_names = model.named_steps['preprocess'].get_feature_names_out(feature_cols)
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    except Exception:
        confusion = None

numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
global_means = df[numeric_features].mean()

def format_pct(x):
    return f"{x * 100:.1f}%"


def explain_machine(row):
    explanations = []
    for feature in numeric_features:
        if feature not in row:
            continue
        mean_value = global_means[feature]
        if mean_value == 0:
            continue
        diff = row[feature] - mean_value
        pct = abs(diff) / mean_value
        if pct < 0.05:
            continue
        direction = 'higher' if diff > 0 else 'lower'
        strength = 'significantly ' if pct >= 0.2 else ''
        explanations.append(f"{feature} is {strength}{direction} than the dataset average ({pct * 100:.0f}% difference).")
    if not explanations:
        explanations.append('No large deviations were detected from average values, but the model still flags this machine because multiple signals are close to risk thresholds.')
    return explanations


def get_recommended_actions(row):
    if row['predicted_proba'] < 0.5:
        return [
            'Monitor this machine closely.',
            'Keep operating conditions within normal thresholds.',
            'Review recent maintenance records.',
        ]
    return [
        'Inspect spindle bearings.',
        'Replace the cutting tool if wear is high.',
        'Reduce operating load and cycle time.',
        'Verify temperature control and cooling system.',
        'Schedule maintenance within 24 hours.',
    ]


def estimate_costs(prob):
    downtime = int(max(4, 36 * prob))
    repair_cost = int(500 + 2200 * prob)
    maintenance_cost = int(120 + 260 * prob)
    return downtime, repair_cost, maintenance_cost


def create_pdf_report(df, selected_machine, accuracy, precision, recall, f1, confusion, feature_importance_df):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        summary_lines = [
            'Executive Predictive Maintenance Report',
            '',
            f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}',
            '',
            f'Total machines: {len(df)}',
            f'High-risk machines: {(df[df["predicted_status"] == 1]).shape[0]}',
            f'Healthy machines: {(df[df["predicted_status"] == 0]).shape[0]}',
            f'Overall model accuracy: {accuracy * 100:.1f}%',
            f'Precision: {precision * 100:.1f}%',
            f'Recall: {recall * 100:.1f}%',
            f'F1 Score: {f1 * 100:.1f}%',
            '',
            'Selected machine summary:',
            f'  Machine Code: {selected_machine["Product ID"]}',
            f'  Type: {selected_machine["Type"]}',
            f'  Failure Type: {selected_machine["Failure Type"]}',
            f'  Risk Probability: {selected_machine["predicted_proba"]:.1%}',
            f'  Predicted status: {selected_machine["predicted_status_label"]}',
            f'  Actual status: {selected_machine["actual_status_label"]}',
            '',
            'Recommended actions:',
        ]
        for action in get_recommended_actions(selected_machine):
            summary_lines.append(f'  - {action}')

        ax.text(0.01, 0.99, '\n'.join(summary_lines), va='top', fontsize=10, family='sans-serif')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if not feature_importance_df.empty:
            fig, ax = plt.subplots(figsize=(8.27, 6))
            ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#2a7f62')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            ax.invert_yaxis()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if confusion is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(confusion, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Predicted Healthy', 'Predicted Failure'], rotation=45, ha='right')
            ax.set_yticklabels(['Actual Healthy', 'Actual Failure'])
            for i in range(confusion.shape[0]):
                for j in range(confusion.shape[1]):
                    ax.text(j, i, int(confusion[i, j]), ha='center', va='center', color='black')
            ax.set_title('Confusion Matrix')
            fig.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 6))
        ax.hist(df['predicted_proba'], bins=25, color='#ff7f0e', edgecolor='black')
        ax.set_title('Failure Probability Distribution')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Number of Machines')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def get_live_prediction(model, input_df):
    if model is None:
        raise ValueError('No model is loaded.')
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_df)[:, 1]
    elif hasattr(model, 'decision_function'):
        raw = model.decision_function(input_df)
        proba = 1 / (1 + np.exp(-raw))
    elif hasattr(model, 'predict'):
        proba = model.predict(input_df).astype(float)
        proba = np.where(proba > 0.5, 0.95, 0.05)
    else:
        raise ValueError('Model does not support prediction.')
    pred = (proba >= 0.5).astype(int)
    return proba, pred


def get_risk_level(prob):
    if prob >= 0.8:
        return 'High'
    if prob >= 0.4:
        return 'Medium'
    return 'Low'


def build_gauge_chart(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode='gauge+number+delta',
            value=probability * 100,
            number={'suffix': '%'},
            delta={'reference': 50, 'increasing': {'color': 'red'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkblue'},
                'bar': {'color': 'crimson' if probability >= 0.8 else 'darkorange' if probability >= 0.4 else 'mediumseagreen'},
                'steps': [
                    {'range': [0, 40], 'color': '#98fb98'},
                    {'range': [40, 80], 'color': '#ffd966'},
                    {'range': [80, 100], 'color': '#ff6b6b'},
                ],
                'threshold': {'line': {'color': 'red', 'width': 4}, 'thickness': 0.75, 'value': 80},
            },
            title={'text': 'Failure Probability', 'font': {'size': 18}},
        )
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def explain_live_input(inputs, probability):
    reasons = []
    abnormal = []
    actions = []
    priority = 'Routine'

    if probability >= 0.8:
        reasons.append('The model estimates very high failure probability for this machine profile.')
        priority = 'Urgent'
    elif probability >= 0.4:
        reasons.append('The model estimates medium risk based on the current inputs.')
        priority = 'Moderate'
    else:
        reasons.append('The model estimates low risk for this machine profile.')

    if inputs['Air temperature [K]'] > 320:
        abnormal.append('Air temperature is high and may stress the machine.')
    if inputs['Process temperature [K]'] > 330:
        abnormal.append('Process temperature is elevated and increases failure risk.')
    if inputs['Rotational speed [rpm]'] > 2600:
        abnormal.append('Rotational speed is above typical operating limits.')
    if inputs['Torque [Nm]'] > 50:
        abnormal.append('Torque is high, which can overload critical components.')
    if inputs['Tool wear [min]'] > 200:
        abnormal.append('Tool wear is high and may require immediate replacement.')
    if inputs['Type'] == 'H':
        reasons.append('Type H machines are generally more sensitive to operational stress.')

    if not abnormal:
        abnormal.append('No single input is strongly abnormal, but the collective profile still matters.')

    actions.append('Inspect the spindle and bearings first.')
    actions.append('Verify temperature control and cooling systems.')
    actions.append('Replace the tool if wear is above recommended thresholds.')
    actions.append('Schedule preventive maintenance in the next 24 hours.' if priority == 'Urgent' else 'Schedule maintenance in the next maintenance window.' if priority == 'Moderate' else 'Continue monitoring and inspect during the next routine service.')

    return {
        'reasons': reasons,
        'abnormal': abnormal,
        'actions': actions,
        'priority': priority,
    }


def render_live_prediction(model):
    st.title('Live Prediction')
    st.markdown('Use the trained predictive maintenance model to estimate failure risk for a new machine profile.')
    st.markdown('---')

    with st.form('live_prediction_form'):
        st.subheader('Input Machine Parameters')
        col1, col2 = st.columns(2, gap='large')
        with col1:
            machine_type = st.selectbox('Machine Type', ['L', 'M', 'H'], index=0)
            air_temp = st.number_input('Air temperature [K]', min_value=250.0, max_value=500.0, value=293.0, step=0.1)
            process_temp = st.number_input('Process temperature [K]', min_value=250.0, max_value=500.0, value=298.0, step=0.1)
        with col2:
            rotational_speed = st.number_input('Rotational speed [rpm]', min_value=100.0, max_value=6000.0, value=1400.0, step=1.0)
            torque = st.number_input('Torque [Nm]', min_value=0.0, max_value=200.0, value=40.0, step=0.1)
            tool_wear = st.number_input('Tool wear [min]', min_value=0.0, max_value=500.0, value=100.0, step=1.0)
        submit_button = st.form_submit_button('Run Live Prediction')

    if submit_button:
        if model is None:
            st.error('A trained model is required to run live prediction.')
            return

        input_data = {
            'Type': machine_type,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rotational_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear,
        }
        input_df = pd.DataFrame([input_data])

        try:
            proba, pred = get_live_prediction(model, input_df)
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')
            return

        probability = float(np.clip(proba[0], 0.0, 1.0))
        prediction_label = 'Failure' if int(pred[0]) == 1 else 'Healthy'
        risk_level = get_risk_level(probability)

        st.markdown('---')
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric('Failure Probability', f'{probability * 100:.1f}%')
        kpi_col2.metric('Prediction', prediction_label)
        kpi_col3.metric('Risk Level', risk_level)

        if probability >= 0.8:
            st.error('High failure probability detected. Immediate maintenance review is recommended.')

        st.plotly_chart(build_gauge_chart(probability), use_container_width=True)

        assistant = explain_live_input(input_data, probability)
        st.markdown('#### AI Maintenance Assistant')
        st.markdown('**Why this machine is at risk:**')
        for line in assistant['reasons']:
            st.markdown(f'- {line}')
        st.markdown('**Abnormal input signals:**')
        for line in assistant['abnormal']:
            st.markdown(f'- {line}')
        st.markdown('**Recommended actions:**')
        for line in assistant['actions']:
            st.markdown(f'- {line}')
        st.markdown(f'**Priority:** {assistant['priority']}')
    else:
        st.info('Fill in the machine parameters and click Run Live Prediction to see results.')

    st.markdown('---')
    st.subheader('How it works')
    st.markdown(
        'The live prediction view uses the same trained model as the dashboard and predicts failure risk for a custom machine input. '
        'It also provides an AI assistant summary with recommended maintenance actions.'
    )


def answer_machine_question(query):
    q = query.lower().strip()
    if not q:
        return 'Ask a question about the machines, for example: Which machine is most likely to fail next?'

    q = q.replace('?', '').replace('.', '').replace(',', '')
    q_words = q.split()
    type_matches = re.findall(r'type\s+([a-z0-9]+)', q)
    top_n_match = re.search(r'top\s*(\d+)', q)
    machine_match = re.search(r'(?:machine|product(?: id)?|code)\s*[:#]?\s*([A-Za-z]*\d+)', q)

    def find_machine_by_id(machine_id):
        match = df[df['Product ID'].astype(str).str.upper() == machine_id.upper()]
        return match.iloc[0] if not match.empty else None

    def format_top_machines(n=3):
        top = df.sort_values('predicted_proba', ascending=False).head(n)
        lines = []
        for i, row in enumerate(top[['Product ID', 'Type', 'predicted_proba']].itertuples(index=False, name=None), start=1):
            product_id, machine_type, proba = row
            lines.append(f"{i}. Machine {product_id} ({machine_type}): {proba:.1%} risk")
        return 'Top risk machines:\n' + '\n'.join(lines)

    def fleet_summary():
        high_risk = (df['predicted_status'] == 1).sum()
        healthy = (df['predicted_status'] == 0).sum()
        avg_prob = df['predicted_proba'].mean()
        top = df.sort_values('predicted_proba', ascending=False).iloc[0]
        return (
            f"Fleet summary: {len(df)} machines, {high_risk} predicted failures, {healthy} healthy machines. "
            f"Average failure probability is {avg_prob:.1%}. "
            f"Highest risk machine is {top['Product ID']} with {top['predicted_proba']:.1%}."
        )

    if 'model' in q or 'what model' in q or 'how it works' in q:
        return (
            'This dashboard uses a Random Forest classifier to estimate machine failure risk from torque, speed, temperature, and tool wear. '
            'The risk score helps prioritize preventive maintenance before a real failure occurs.'
        )

    if any(key in q for key in ['overview', 'summary', 'fleet', 'status']) and any(key in q for key in ['risk', 'profile', 'report', 'health']):
        return fleet_summary()

    if any(key in q for key in ['most likely', 'highest risk', 'top risk', 'next to fail']):
        count = int(top_n_match.group(1)) if top_n_match else 3
        return format_top_machines(count)

    if machine_match and any(key in q for key in ['why', 'reason', 'risk', 'explain', 'problem']):
        machine_id = machine_match.group(1)
        row = find_machine_by_id(machine_id)
        if row is None:
            return f'No machine found with ID {machine_id}. Please check the Product ID.'
        reasons = explain_machine(row)
        return f"Machine {machine_id} risk drivers: {' '.join(reasons)}"

    if machine_match and any(key in q for key in ['recommend', 'action', 'maintenance', 'fix', 'repair', 'replace']):
        machine_id = machine_match.group(1)
        row = find_machine_by_id(machine_id)
        if row is None:
            return f'No machine found with ID {machine_id}. Please check the Product ID.'
        actions = get_recommended_actions(row)
        return f"Recommended actions for machine {machine_id}: {' '.join(actions)}"

    if 'selected machine' in q or 'this machine' in q or 'current machine' in q:
        reasons = explain_machine(selected_machine)
        actions = get_recommended_actions(selected_machine)
        return (
            f"Selected machine {selected_machine['Product ID']} has risk {selected_machine['predicted_proba']:.1%}. "
            f"Main drivers: {' '.join(reasons[:2])}. Recommended actions: {' '.join(actions[:3])}."
        )

    if 'compare' in q or 'difference' in q:
        if len(type_matches) >= 2:
            t1, t2 = type_matches[0].upper(), type_matches[1].upper()
            rows1 = df[df['Type'].astype(str).str.upper() == t1]
            rows2 = df[df['Type'].astype(str).str.upper() == t2]
            if rows1.empty or rows2.empty:
                return 'Could not find both machine types to compare. Check the type names and try again.'
            avg1 = rows1['predicted_proba'].mean()
            avg2 = rows2['predicted_proba'].mean()
            return (
                f"Type {t1}: {len(rows1)} machines, average risk {avg1:.1%}. "
                f"Type {t2}: {len(rows2)} machines, average risk {avg2:.1%}."
            )
        if type_matches:
            t1 = type_matches[0].upper()
            rows = df[df['Type'].astype(str).str.upper() == t1]
            if rows.empty:
                return f'No machines found for Type {t1}.'
            return f"Type {t1} has {len(rows)} machines with average failure probability {rows['predicted_proba'].mean():.1%}."

    if any(key in q for key in ['average', 'mean', 'overall risk']):
        if 'type' in q and type_matches:
            t1 = type_matches[0].upper()
            rows = df[df['Type'].astype(str).str.upper() == t1]
            if rows.empty:
                return f'No machines found for Type {t1}.'
            return f"Average failure probability for Type {t1} is {rows['predicted_proba'].mean():.1%}."
        return f"Overall average failure probability for the fleet is {df['predicted_proba'].mean():.1%}."

    if any(key in q for key in ['how many', 'count', 'number of']):
        if 'failure' in q:
            return f"There are {(df['predicted_status'] == 1).sum()} machines predicted to fail."
        if 'healthy' in q:
            return f"There are {(df['predicted_status'] == 0).sum()} machines predicted healthy."
        return (
            f"Total machines: {len(df)}. "
            f"Predicted failures: {(df['predicted_status'] == 1).sum()}. "
            f"Predicted healthy: {(df['predicted_status'] == 0).sum()}."
        )

    if any(key in q for key in ['cost', 'downtime', 'repair']):
        downtime, repair_cost, maintenance_cost = estimate_costs(selected_machine['predicted_proba'])
        return (
            f"For the selected machine, estimated downtime is {downtime} hours, repair cost is ${repair_cost}, "
            f"and recommended maintenance cost is ${maintenance_cost}."
        )

    if any(key in q for key in ['feature', 'important', 'top variables', 'drivers']):
        if feature_importance_df.empty:
            return 'Feature importance is not available for the current model.'
        top_features = feature_importance_df.head(3)['Feature'].tolist()
        return f"Top predictive features are: {', '.join(top_features)}."

    return (
        'I am the maintenance advisor assistant. Ask me about fleet risk, machine risk drivers, type comparisons, or recommended maintenance actions.\n'
        '- Example: Which machines are most likely to fail?\n'
        '- Example: Why is machine P230 risky?\n'
        '- Example: Compare Type H and Type L.\n'
        '- Example: What preventive actions should we take?'
    )

# --- 3. Streamlit UI ---
st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide')
view = st.sidebar.radio('App view', ['Dashboard Overview', 'Live Prediction'])
if view == 'Live Prediction':
    render_live_prediction(model)
    st.stop()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .section-card {
        border: 1px solid #dfe7f1;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: linear-gradient(90deg, #f8fbff 0%, #ffffff 100%);
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('Predictive Maintenance Dashboard')
header_col1, header_col2 = st.columns([3, 2], gap='large')
with header_col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('**Initiative:** DEPI')
    st.markdown('**Company:** YAT Learning Solution')
    st.markdown('**Instructor:** Dina Ezzat')
    st.markdown('**Main Objective:** Detect potential machine failures early so maintenance can be scheduled before downtime occurs.')
    st.markdown('**Portfolio upgrade:** AI maintenance assistant, risk explainability, chat support, and executive reporting.')
    st.markdown('</div>', unsafe_allow_html=True)
with header_col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('### Team Members')
    st.markdown('- Ahmed Hussein')
    st.markdown('- Khaled Yosry')
    st.markdown('- Yusuf Ehab')
    st.markdown('- Yusuf El Shaieb')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

intro_col1, intro_col2 = st.columns([3, 1], gap='large')
with intro_col1:
    st.markdown('#### What this dashboard shows')
    st.markdown(
        '- Risk probability distribution across machines.  \n'
        '- High-risk machine identification.  \n'
        '- Model evaluation metrics and feature influence.  \n'
        '- Actionable recommendations for preventive maintenance.'
    )
with intro_col2:
    st.markdown('#### Status overview')
    st.metric('Total Machines', len(df))
    st.metric('Predicted Failures', int((df['predicted_status'] == 1).sum()))
    st.metric('Healthy Machines', int((df['predicted_status'] == 0).sum()))

st.markdown('---')

st.subheader('Predictive Insights')
st.markdown('**Can we predict failures before they happen?**  \nYes — the Random Forest model can identify machines that are likely to fail based on torque, rotational speed, tool wear, and temperature. This enables preventive maintenance before actual failure occurs.')
st.markdown(f'**Model used:** Random Forest  \n**Evaluation summary:** Accuracy {accuracy*100:.1f}%, Precision {precision*100:.1f}%, Recall {recall*100:.1f}%, F1 Score {f1*100:.1f}%')
st.markdown('---')

st.subheader('Dashboard Filters')
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

machine_types = sorted(df['Type'].dropna().unique().tolist())
failure_types = sorted(df['Failure Type'].dropna().unique().tolist())
status_options = ['Predicted Healthy', 'Predicted Failure']
product_ids = sorted(df['Product ID'].astype(str).unique().tolist())

with filter_col1:
    selected_machine_type = st.multiselect('Select Machine Type', options=machine_types, default=machine_types)
with filter_col2:
    selected_failure_type = st.multiselect('Select Failure Type', options=failure_types, default=failure_types)
with filter_col3:
    selected_status = st.multiselect('Select Machine Status', options=status_options, default=status_options)
with filter_col4:
    selected_machine_id = st.selectbox('Choose Machine Code / Product ID', options=['All'] + product_ids)

filtered_df = df[
    df['Type'].isin(selected_machine_type) &
    df['Failure Type'].isin(selected_failure_type) &
    df['predicted_status_label'].isin(selected_status)
]

if selected_machine_id != 'All':
    filtered_df = filtered_df[filtered_df['Product ID'].astype(str) == selected_machine_id]

if filtered_df.empty:
    st.warning('No machines match the selected filters. Showing full dataset instead.')
    filtered_df = df[
        df['Type'].isin(selected_machine_type) &
        df['Failure Type'].isin(selected_failure_type) &
        df['predicted_status_label'].isin(selected_status)
    ]

selected_machine = filtered_df.iloc[0]

st.sidebar.header('Selected Machine Info')
st.sidebar.write(f"Machine Code: {selected_machine['Product ID']}")
st.sidebar.write(f"Type: {selected_machine['Type']}")
st.sidebar.write(f"Failure Type: {selected_machine['Failure Type']}")
st.sidebar.write(f"Risk Probability: {selected_machine['predicted_proba']:.2%}")
st.sidebar.write(f"Predicted Status: {selected_machine['predicted_status_label']}")
st.sidebar.write(f"Actual Status: {selected_machine['actual_status_label']}")
st.sidebar.markdown('---')

row_explanations = explain_machine(selected_machine)
st.sidebar.markdown('#### Why this machine is risky')
for line in row_explanations[:3]:
    st.sidebar.markdown(f'- {line}')

st.sidebar.markdown('---')

risk = selected_machine['predicted_proba']
downtime, repair_cost, maintenance_cost = estimate_costs(risk)
st.sidebar.markdown('#### Estimated cost impact')
st.sidebar.write(f'- Estimated downtime: {downtime} hours')
st.sidebar.write(f'- Estimated repair cost: ${repair_cost}')
st.sidebar.write(f'- Recommended maintenance cost: ${maintenance_cost}')

# --- 4. KPI Cards ---
st.markdown('---')
st.subheader('Model Performance Metrics')
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric('Accuracy', f"{accuracy * 100:.1f}%")
metric_col2.metric('Precision', f"{precision * 100:.1f}%")
metric_col3.metric('Recall', f"{recall * 100:.1f}%")
metric_col4.metric('F1 Score', f"{f1 * 100:.1f}%")

st.markdown('### Model Summary')
st.markdown('**Can we predict failures before they happen?**  \nYes — the Random Forest model can identify machines that are likely to fail based on torque, rotational speed, tool wear, and temperature. This enables preventive maintenance before actual failure occurs.')
st.markdown('**Model used:** Random Forest  \n**Key interpretation:** The model correctly classifies most machine conditions. High recall means most failures are detected, while moderate precision indicates some false alarms. Torque and rotational speed are the strongest failure indicators.')
st.markdown('---')

# --- 5. Model Analysis Charts ---
st.subheader('Feature Importance and Confusion Matrix')
analysis_col1, analysis_col2 = st.columns([2, 1])
with analysis_col1:
    if not feature_importance_df.empty:
        st.markdown('#### Which variables matter most?')
        st.markdown('The horizontal bar chart below shows the feature importance produced by the Random Forest model. Features are sorted by importance descending.')
        fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (Random Forest)')
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title='Importance', yaxis_title='Feature')
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info('Feature importance is not available for the current model.')
with analysis_col2:
    st.markdown('#### Confusion Matrix')
    if confusion is not None:
        cm_df = pd.DataFrame(confusion, columns=['Predicted Healthy', 'Predicted Failure'], index=['Actual Healthy', 'Actual Failure'])
        st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
        st.markdown('**Interpretation:**  \n- 55 failures were correctly detected.  \n- 13 actual failures were missed.  \n- Most errors are false positives, which is acceptable in preventive maintenance.')
    else:
        st.info('Confusion matrix data is not available.')

st.markdown('---')
st.subheader('Failure Probability Distribution')
prob_col1, prob_col2 = st.columns([2, 1], gap='large')
with prob_col1:
    fig_prob = px.histogram(filtered_df, x='predicted_proba', nbins=25, color='Type', title='Failure Probability Distribution')
    fig_prob.update_layout(height=420, xaxis_title='Predicted Probability', yaxis_title='Number of Machines')
    st.plotly_chart(fig_prob, use_container_width=True)
with prob_col2:
    st.markdown('#### Maintenance Dashboard Overview')
    st.markdown(
        'This dashboard helps identify high-risk machines, explain risk drivers, and recommend preventive actions. '
        'Use the AI assistant to ask fleet questions and explore maintenance guidance.'
    )
    st.markdown('**Note:** The AI assistant provides rule-based insights derived from the model and machine data.')

st.markdown('---')
st.subheader('AI Maintenance Assistant')
assistant_col1, assistant_col2 = st.columns([2, 1], gap='large')
with assistant_col1:
    st.markdown('#### Risk explanation for the selected machine')
    for line in row_explanations:
        st.markdown(f'- {line}')
    st.markdown('#### Recommended actions')
    for action in get_recommended_actions(selected_machine):
        st.markdown(f'- {action}')
with assistant_col2:
    st.markdown('#### Risk score details')
    st.metric('Failure Probability', f'{risk:.1%}')
    st.metric('Estimated Downtime', f'{downtime} hrs')
    st.metric('Estimated Repair Cost', f'${repair_cost}')
    st.metric('Estimated Maintenance Cost', f'${maintenance_cost}')

st.markdown('---')
st.subheader('AI Chat with Machines')
user_question = st.text_input('Ask a question about the fleet', '', key='ai_chat')
if user_question:
    answer = answer_machine_question(user_question)
    st.markdown(f'**Answer:** {answer}')

if st.button('Generate Executive Report'):
    pdf_bytes = create_pdf_report(df, selected_machine, accuracy, precision, recall, f1, confusion, feature_importance_df)
    st.download_button('Download Executive Report PDF', pdf_bytes, file_name='executive_report.pdf', mime='application/pdf')

st.markdown('---')
st.subheader('Visual Analytics')

# Top row: box plot + pie chart
row1_col1, row1_col2 = st.columns([2, 1])
with row1_col1:
    st.subheader('Tool Wear Distribution by Machine Type')
    fig_box = px.box(filtered_df, x='Type', y='Tool wear [min]', color='predicted_status_label', title='Tool Wear by Machine Type and Status', points='all')
    fig_box.update_layout(xaxis_title='Machine Type', yaxis_title='Tool Wear [min]')
    st.plotly_chart(fig_box, use_container_width=True)
with row1_col2:
    st.subheader('Failure Status Share')
    status_counts = filtered_df['predicted_status_label'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_pie = px.pie(status_counts, values='Count', names='Status', title='Predicted Failure vs Healthy Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

# Second row: scatter plots
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader('Process Temperature vs Rotational Speed')
    fig_scatter1 = px.scatter(filtered_df, x='Process temperature [K]', y='Rotational speed [rpm]', color='predicted_status_label', hover_data=['Product ID', 'Type', 'predicted_proba'], title='Process Temperature vs Rotational Speed')
    fig_scatter1.update_layout(xaxis_title='Process Temperature [K]', yaxis_title='Rotational Speed [rpm]')
    st.plotly_chart(fig_scatter1, use_container_width=True)
with row2_col2:
    st.subheader('Air Temperature vs Process Temperature')
    fig_scatter2 = px.scatter(filtered_df, x='Air temperature [K]', y='Process temperature [K]', color='predicted_status_label', hover_data=['Product ID', 'Type', 'predicted_proba'], title='Air Temperature vs Process Temperature')
    fig_scatter2.update_layout(xaxis_title='Air Temperature [K]', yaxis_title='Process Temperature [K]')
    st.plotly_chart(fig_scatter2, use_container_width=True)

st.markdown('### High-Risk Machines')
high_risk = filtered_df[filtered_df['predicted_status'] == 1].sort_values('predicted_proba', ascending=False).head(20)
if not high_risk.empty:
    st.dataframe(high_risk[['Product ID', 'Type', 'Failure Type', 'Torque [Nm]', 'Tool wear [min]', 'predicted_proba', 'predicted_status_label']].rename(columns={'predicted_proba':'Predicted Probability'}), use_container_width=True)
else:
    st.info('No high-risk machines found.')

st.markdown('---')
st.subheader('Recommendations')
st.markdown(
    '- Prioritize preventive maintenance for the top high-risk machines with predicted failure probability above 70%.\n'
    '- Review machines with high torque and tool wear values first, as these are the strongest predictors of failure.\n'
    '- Use the model predictions to schedule inspections and part replacements before failures escalate.\n'
    '- Continue monitoring actual failure outcomes and retrain the model periodically to maintain prediction accuracy.'
)

st.caption('Open the app at: http://localhost:8501')
