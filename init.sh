sudo python3 -m pip install --upgrade pip
python3 -m venv myenv --without-pip
source /myenv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
# myenv/bin/activate
pip install -r requirements.txt
# put "deactivate" on the terminal to deactivate the virtual enviroment
