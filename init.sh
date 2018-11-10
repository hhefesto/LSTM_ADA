# use python 3.6. python 3.7 crashed on windows

sudo python3 -m pip install --upgrade pip
python3 -m venv myenv
# python3 -m venv myenv --without-pip
# Set-ExecutionPolicy Unrestricted # esto sólo se usa en windows
source /myenv/bin/activate # esto se tiene que hacer con windows también, pero que no funciona source
#.\myenv\Scripts\activate # lo mismo pero para windows
# curl https://bootstrap.pypa.io/get-pip.py | python # esto va con lo de --without-pip
pip install -r requirements.txt
# put "deactivate" on the terminal to deactivate the virtual enviroment
