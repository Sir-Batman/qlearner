from setuptools import setup

setup(name="QLearner",
      version="0.1",
      description="A basic Q-Learner class.",
      url="https://github.com/Sir-Batman/QLearner",
      author="Connor Yates",
      author_email="yatesco@oregonstate.edu",
      license="MIT",
      packages=["qlearner"],
      install_requires=[
          "numpy"
      ],
      zip_safe=False)
