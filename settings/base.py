from pathlib import Path

from environs import Env


env = Env()

STAND = env.str('STAND', default='local')
BASE_PATH = Path.cwd().absolute()

if STAND == 'local':
    env.read_env(path=str(BASE_PATH / '.env'))
