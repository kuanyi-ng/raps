"""
Module to capture Account classes Classes:
- Account: representation of an Account
- Accounts: collection of all accounts
"""
import json
from .job import JobStatistics


class Account:
    """Represents an account of a user.

    Each users holds attributes for accounting and statistics, which is used
    for summaries
    Each job consists of various attributes such as the number of nodes required for execution,
    CPU and GPU utilization, wall time, and other relevant parameters (see utils.job_dict).
    The job can transition through different states during its lifecycle, including PENDING,
    RUNNING, COMPLETED, CANCELLED, FAILED, or TIMEOUT.
    """

    def __init__(self, id, name,
                 priority=0,
                 total_jobs_enqueued=0,
                 total_jobs_completed=0,
                 time_allocated=0,
                 energy_allocated=0,
                 avg_power=0,
                 fugaku_points=0
                 ):
        self.id = id
        self.name = name
        self.priority = priority
        self.total_jobs_enqueued = total_jobs_enqueued
        self.total_jobs_completed = total_jobs_completed
        self.time_allocated = time_allocated
        self.energy_allocated = energy_allocated
        self.avg_power = avg_power
        if self.avg_power == 0 and self.energy_allocated != 0:
            self.avg_power = self.time_allocated / self.energy_allocated
        self.fugaku_points = fugaku_points

    def update_statistics(self, jobstats, average_user):
        self.total_jobs_completed += 1
        self.time_allocated += jobstats.run_time
        self.energy_allocated += jobstats.energy
        if self.time_allocated == 0:
            self.avg_power = 0
        else:
            self.avg_power = self.energy_allocated / self.time_allocated
        if average_user.avg_power == 0:  # If this is the first job use own power
            average_user.avg_power = self.avg_power
        if average_user.avg_power != 0:  # If no energy was computed no points can be computed.
            self.fugaku_points = (average_user.energy_allocated - self.energy_allocated) / average_user.avg_power

    def __repr__(self):
        return (f"Account(id={self.id}, name={self.name}), "
                f"priority: {self.priority}, "
                f"total_jobs_enqueued: {self.total_jobs_enqueued}, "
                f"total_jobs_completed: {self.total_jobs_completed}, "
                f"time_allocated: {self.time_allocated}, "
                f"energy_allocated: {self.energy_allocated}, "
                f"avg_power: {self.avg_power}, "
                f"fugaku_points: {self.fugaku_points}, "
                )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "total_jobs_enqueued": self.total_jobs_enqueued,
            "total_jobs_completed": self.total_jobs_completed,
            "time_allocated": self.time_allocated,
            "energy_allocated": self.energy_allocated,
            "avg_power": self.avg_power,
            "fugaku_points": self.fugaku_points
        }

    @classmethod
    def init_from_dict(acct, account_dict):  # id ,name, priority, total_jobs, time_allocated, energy_allocated, avg_power, fugaku_points):
        acct = Account(account_dict["id"], account_dict["name"], priority=account_dict["priority"])
        acct.id = account_dict["id"]
        acct.name = account_dict["name"]
        acct.priority = account_dict["priority"]
        acct.total_jobs = account_dict["total_jobs"]
        acct.time_allocated = account_dict["time_allocated"]
        acct.energy_allocated = account_dict["energy_allocated"]
        acct.avg_power = account_dict["avg_power"]
        acct.fugaku_points = account_dict["fugaku_points"]
        return acct


class Accounts:

    def update_average_user(self):
        total_accounts = len(self.account_dict)
        self.average_user.total_jobs_enqueued = self.all_users.total_jobs_enqueued / total_accounts
        self.average_user.total_jobs_completed = self.all_users.total_jobs_completed / total_accounts
        self.average_user.time_allocated = self.all_users.time_allocated / total_accounts
        self.average_user.energy_allocated = self.all_users.energy_allocated / total_accounts
        self.average_user.avg_power = self.all_users.avg_power / total_accounts
        self.fugaku_points = self.all_users.fugaku_points / total_accounts  # this should be 0
        return self

    def __init__(self, jobs=None):
        self._account_id = 0
        self.account_dict = dict()
        self.all_users = Account(-2, "All_Users")
        self.average_user = Account(-1, "Avg_User")
        if jobs:
            if not isinstance(jobs,list):
                raise TypeError
            for job_dict in jobs:
                if not isinstance(job_dict,dict):
                    raise TypeError
                if job_dict["account"] not in self.account_dict:
                    self.account_dict[job_dict["account"]] = Account(self._account_id, job_dict["account"], total_jobs_enqueued=1)
                    self._account_id += 1
                else:
                    self.account_dict[job_dict["account"]].total_jobs_enqueued += 1

    def initialize_accounts_from_dict(self, dictionary):
        if '_account_id' in dictionary:
            self._account_id = dictionary['_account_id']
        if 'account_dict' in dictionary:
            dics_from_dictionary = dictionary['account_dict']
            self.account_dict = {}
            for account_name, account_dict in dics_from_dictionary.items():
                self.account_dict[account_name] = Account.init_from_dict(account_dict)

        if 'all_users' in dictionary:
            self.all_users = Account.init_from_dict(dictionary['all_users'])
        if 'average_user' in dictionary:
            self.average_user = Account.init_from_dict(dictionary['average_user'])

    def initialize_accounts_from_json(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                json_object = json.load(file)
            self.initialize_accounts_from_dict(json_object)
        except ValueError:
            raise ValueError(f"{file} could not be read using json.load()")

    def update_account_statistics(self, jobstats):
        # Update specific account associated with job
        if isinstance(jobstats, JobStatistics):
            if jobstats.account not in self.account_dict:
                raise ValueError(f"Account {jobstats.account} not registered in Accounts object {self}")
                # self.account_dict[jobstats.account] = Account(self._account_id, jobstats.account)
                # self._account_id += 1
            account = self.account_dict[jobstats.account]
            account.update_statistics(jobstats, self.average_user)
            self.account_dict[jobstats.account] = account
            # Update the  summary account (all_users) and the average_user account
            self.all_users.update_statistics(jobstats,self.average_user)
            self.update_average_user()

    def to_dict(self):
        acct_dict = {}
        for account_name,account in self.account_dict.items():
            acct_dict[account_name] = account.to_dict()
        ret_dict = {}
        ret_dict['_account_id'] = self._account_id
        ret_dict['account_dict'] = acct_dict
        ret_dict['all_users'] = self.all_users.to_dict()
        ret_dict['average_user'] = self.average_user.to_dict()
        return ret_dict
