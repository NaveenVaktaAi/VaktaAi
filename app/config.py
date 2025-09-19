# import os

# from dotenv import dotenv_values


# def env_variables():
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     mode = os.getenv("MODE", "local")

#     if mode == "production":
#         env_file = os.path.join(current_directory, "../.env.production")
#     elif mode == "development":
#         env_file = os.path.join(current_directory, "../.env.development")
#     elif mode == "uat":
#         env_file = os.path.join(current_directory, "../.env.uat")
#     else:
#         env_file = os.path.join(current_directory, "../.env.local")
#     # if mode == "production" else "../.env.development"
#     env = dotenv_values(env_file)

#     return env
