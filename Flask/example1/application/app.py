from flask import Flask, render_template, jsonify, Response
import pandas as pd
import json

app = Flask(__name__)

# Reading symbols from the file symbols.txt
with open('symbols.txt', 'r') as read_file:
    response = read_file.read()
    read_file.close()

# Process format as JSON
response_json = json.loads(response)

# Create a DataFrame from the response
symbols_df = pd.DataFrame(response_json)

# getting rid of inactive currencies
symbols_df['highPrice'] = symbols_df['highPrice'].astype('float32')
symbols_df_noChange = symbols_df[symbols_df.highPrice.values == 0]
symbols_df = symbols_df.drop(symbols_df_noChange.index, axis=0)
symbols_df.sort_values(by = 'symbol', ascending=True, inplace=True)
symbols_df = symbols_df.reset_index()
symbols_df = symbols_df.drop(['index'], axis=1)

# Extract the symbols
symbols = symbols_df.symbol.values

# home page
@app.route('/')
def home():
    return render_template('home.html')

# Symbols page - click on each symbol to get a stats report
@app.route('/symbols', methods=['GET'])
def get_symbols():
    return render_template('symbols.html', symbols = symbols)

# Stats report webpage
@app.route('/symbols/<symbol_pair>', methods=['GET'])
def get_symbol_report(symbol_pair):
    
    # Grab the specifc symbol from the symbols DataFrame
    symbol_df = symbols_df[symbols_df.symbol == symbol_pair]

    # Extract symbol stats
    symbol = symbol_df.symbol.values
    open = symbol_df.openPrice.values
    close = symbol_df.lastPrice.values
    high = symbol_df.highPrice.values
    low = symbol_df.lowPrice.values
    volume = symbol_df.volume.values

    return render_template('symbol.html', symbol = symbol, high = high, low = low, open = open, close = close, volume = volume)

# API for extracting all symbols - response in JSON 
@app.route('/api.symbols/api_key=<key>', methods=['GET'])
def get_symbols_api_json(key):
    if key == '12345':
        return response
    else:
        raise PermissionError('Wrong API Key - Authentication Failed!.')


# API for extracting all symbols - response in CSV 
@app.route('/api.symbols/api_key=<api_key>.csv', methods=['GET'])
def get_symbols_api_csv(api_key):
    if api_key == '12345':        
        return symbols_df.to_csv(index=False)
    else:
        raise PermissionError('Wrong API Key - Authentication Failed!.')

# API for extracting a number of symbols
@app.route('/api.symbols/api_key=<api_key>&&number_of_symbols=<number_of_symbols>', methods=['GET'])
def get_symbols_api_json_batch(api_key, number_of_symbols):

    if api_key == '12345':
        return response_json[0:int(number_of_symbols)]
    else:
        raise PermissionError('Wrong API Key - Authentication Failed!.')
    

# API for extracting a number of symbols - response in CSV 
@app.route('/api.symbols/api_key=<api_key>&&number_of_symbols=<number_of_symbols>.csv', methods=['GET'])
def get_symbols_api_csv_batch(api_key, number_of_symbols):
    if api_key == '12345':
        return symbols_df.iloc[0:int(number_of_symbols), :].to_csv(index=False)
    else:
        raise PermissionError('Wrong API Key - Authentication Failed!.')
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
