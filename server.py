from flask import Flask, request, jsonify

app = Flask(__name__)

# API for running scenarios
@app.route('/run', methods=['POST'])
def run_scenarios():

    # File must be a CSV
    max_c_rate = request.form.get('max_c_rate')
    pack_energy_cap = request.form.get('pack_energy_cap')
    pack_max_ah = request.form.get('pack_max_ah')
    pack_max_voltage = request.form.get('pack_max_voltage')
    pack_voltage = request.form.get('pack_voltage')
    soh = request.form.get('SOH')
    soc = request.form.get('SOC')
    start_year = request.form.get('start_year')
    start_month = request.form.get('start_month')
    
    '''
    file = request.files.get('file')
    if file:
        print(file.filename)
        # save the file and the data path
        data_path = 'SolarData/temp.csv'
        save_path = os.path.join(current_dir, '../EV50_cosimulation', data_path)
        file.save(save_path)
    '''

    # Make scenarios
    # Run scenarios

    return jsonify({'message': 'Success'})

if __name__ == '__main__':
    app.run(debug=True, port=4000) # Set debug=False when no longer in development