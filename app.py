from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import pandas as pd
import joblib
import os
import numpy as np
import traceback
import io

from collections import Counter
from datetime import datetime



last_comparison_results = {}

HISTORY_FILE = 'prediction_history.csv'

if not os.path.exists(HISTORY_FILE):

    history_df = pd.DataFrame(columns=[

        'timestamp',
        'model',

        'normal',
        'dos',
        'probe',
        'r2l',
        'u2r',
        'other',

        'accuracy',
        'total_rows'

    ])

    history_df.to_csv(HISTORY_FILE, index=False)



app = Flask(__name__)

app.secret_key = 'nsl-kdd-ids-2026-secret-key-change-in-production'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



FEATURE_COLS = [

    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'

]

CATEGORICAL_COLS = [

    'protocol_type',
    'service',
    'flag'

]



models = {}

model_results = {}

try:

    if os.path.exists('model/random_forest.pkl'):

        models['random_forest'] = joblib.load(
            'model/random_forest.pkl'
        )

    if os.path.exists('model/logistic.pkl'):

        models['logistic'] = joblib.load(
            'model/logistic.pkl'
        )

    if os.path.exists('model/results.pkl'):

        results_file = joblib.load(
            'model/results.pkl'
        )

        for name in models.keys():

            if name in results_file:

                model_results[name] = results_file[name]

            else:

                model_results[name] = {
                    'accuracy': 0.0
                }

    else:

        for name in models.keys():

            model_results[name] = {
                'accuracy': 0.0
            }

    print("Models loaded:", list(models.keys()))

    print("Results loaded:", model_results)

except Exception as e:

    print(f"Error loading models: {e}")

    models = {}

    model_results = {}



@app.route('/')
def index():

    return render_template(

        'index.html',

        model_results=model_results,

        model_names=list(models.keys())

    )



@app.route('/predict', methods=['POST'])
def predict():

    global last_comparison_results

    try:

        # Check upload
        if 'file' not in request.files:

            flash('No file uploaded!')

            return redirect(url_for('index'))

        file = request.files['file']

        if file.filename == '':

            flash('No file selected!')

            return redirect(url_for('index'))

     
        df = pd.read_csv(file)

        print(f"Uploaded rows: {len(df)}")

        
        missing_cols = [

            col for col in FEATURE_COLS

            if col not in df.columns

        ]

        if missing_cols:

            msg = f"Missing required columns: {', '.join(missing_cols[:5])}"

            if len(missing_cols) > 5:
                msg += "..."

            flash(msg)

            return redirect(url_for('index'))

        df = df[FEATURE_COLS].copy()

       
        for col in CATEGORICAL_COLS:

            df[col] = (

                df[col]
                .astype(str)
                .str.strip()
                .str.lower()

            )

        
        numeric_cols = [

            col for col in FEATURE_COLS

            if col not in CATEGORICAL_COLS

        ]

        for col in numeric_cols:

            df[col] = pd.to_numeric(

                df[col],

                errors='coerce'

            ).fillna(0)


        comparison_results = {}

        for model_name, model in models.items():

            
            preds = model.predict(df)

            probs = (

                model.predict_proba(df)[:, 1]

                if hasattr(model, 'predict_proba')

                else None

            )

            attack_counts = Counter(preds)

            total_attacks = (

                attack_counts.get('DoS', 0)

                + attack_counts.get('Probe', 0)

                + attack_counts.get('R2L', 0)

                + attack_counts.get('U2R', 0)

                + attack_counts.get('Other', 0)

            )

            anomaly_percent = (

                (total_attacks / len(df)) * 100

                if len(df) > 0 else 0

            )

            
            comparison_results[model_name] = {

                'model_name': model_name,

                'normal': attack_counts.get('Normal', 0),

                'dos': attack_counts.get('DoS', 0),

                'probe': attack_counts.get('Probe', 0),

                'r2l': attack_counts.get('R2L', 0),

                'u2r': attack_counts.get('U2R', 0),

                'other': attack_counts.get('Other', 0),

                'anomaly': total_attacks,

                'total_rows': len(df),

                'anomaly_percent': round(
                    anomaly_percent,
                    2
                ),

                'accuracy': model_results.get(
                    model_name,
                    {}
                ).get(
                    'accuracy',
                    0.0
                ),

                'avg_anomaly_prob': (

                    f"{np.mean(probs):.1%}"

                    if probs is not None

                    else "N/A"

                )

            }


            history_entry = {

                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),

                'model': model_name,

                'normal': attack_counts.get('Normal', 0),

                'dos': attack_counts.get('DoS', 0),

                'probe': attack_counts.get('Probe', 0),

                'r2l': attack_counts.get('R2L', 0),

                'u2r': attack_counts.get('U2R', 0),

                'other': attack_counts.get('Other', 0),

                'accuracy': round(

                    model_results.get(
                        model_name,
                        {}
                    ).get(
                        'accuracy',
                        0.0
                    ) * 100,

                    2

                ),

                'total_rows': len(df)

            }

            history_df = pd.read_csv(HISTORY_FILE)

            history_df = pd.concat(

                [

                    history_df,

                    pd.DataFrame([history_entry])

                ],

                ignore_index=True

            )

            history_df.to_csv(
                HISTORY_FILE,
                index=False
            )

        
        df_sample = df.head(10).round(2).to_html(

            classes='table table-striped table-bordered table-sm',

            index=False

        )

       
        last_comparison_results = comparison_results

        
        return render_template(

            'results.html',

            predictions=comparison_results,

            total_rows=len(df),

            df_sample=df_sample

        )

    except Exception as e:

        print(f"Prediction error: {e}")

        print(traceback.format_exc())

        flash(f"Prediction failed: {str(e)}")

        return redirect(url_for('index'))


@app.route('/history')
def history():

    if not os.path.exists(HISTORY_FILE):

        flash('No prediction history found.')

        return redirect(url_for('index'))

    history_df = pd.read_csv(HISTORY_FILE)

    history_table = history_df.iloc[::-1].to_html(

        classes='''
        min-w-full
        text-sm
        text-left
        border-separate
        border-spacing-y-2
        ''',

        index=False,

        border=0

    )

    return render_template(

        'history.html',

        history_table=history_table

    )


@app.route('/download-report')
def download_report():

    global last_comparison_results

    if not last_comparison_results:

        flash(
            'No report available yet. Please run prediction first.'
        )

        return redirect(url_for('index'))

    report_df = pd.DataFrame([

        {

            'Model': result['model_name'],

            'Normal': result['normal'],

            'DoS': result['dos'],

            'Probe': result['probe'],

            'R2L': result['r2l'],

            

            'Other': result['other'],

            'Total Attacks': result['anomaly'],

            'Anomaly %': result['anomaly_percent'],

            'Accuracy %': round(
                result['accuracy'] * 100,
                2
            ),

            'Avg Anomaly Probability':
                result['avg_anomaly_prob']

        }

        for result in last_comparison_results.values()

    ])

    output = io.StringIO()

    report_df.to_csv(
        output,
        index=False
    )

    output.seek(0)

    mem = io.BytesIO()

    mem.write(
        output.getvalue().encode('utf-8')
    )

    mem.seek(0)

    return send_file(

        mem,

        mimetype='text/csv',

        as_attachment=True,

        download_name='model_comparison_report.csv'

    )



if __name__ == '__main__':

    os.makedirs(
        'templates',
        exist_ok=True
    )

    os.makedirs(
        'static',
        exist_ok=True
    )

    print("NSL-KDD Intrusion Detection System")

    print("Available models:", list(models.keys()))

    print("Visit: http://127.0.0.1:5000")

    app.run(

        debug=True,

        host='127.0.0.1',

        port=5000

    )