# Vercel Serverless Entrypoint
from weather_app import app

# Vercel uses the 'app' object imported above to serve requests.
if __name__ == '__main__':
    app.run()
