# This module creates a deployment that retrains the model every week.

from flow import train_model_flow

if __name__ == "__main__":
    train_model_flow.serve(name="weekly-retraining", cron="0 0 * * 0")
