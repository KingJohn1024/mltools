import tqdm
import pandas as pd
import multiprocessing.pool as mpp
from functools import partial
from typing import List
from multiprocessing import cpu_count


def istarmap(self, func, iterable, chunksize=1) -> tuple:
    """starmap-version of imap"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


def parallel_process_function(func, partition_by, **kwargs) -> pd.DataFrame:
    """
    Generalized local parallel processing - consuming half of the locally available CPUs by default
    The first parameter of func will be mapped to the values in the partition_by list, additional kwargs are passed

    :param func: Function to call over partitions of the data
    :param partition_by: List of elements to partition by or list of tuples (partitions with data)
    :param kwargs: Key word arguments to pass through to the function "func"
        optional kwargs with explicit processing:
        - axis: control what axis to use when concatenating the partitioned frames, defaults to zero
        - loading_desc: passed to tqdm to include a verbose description in the progress bar

    Example usage:
    >>> parallel_process_function(func=test_func, partition_by=[d1, d2, d3], df=df, parm1=parm)
    >>> parallel_process_function(func=test_func2, partition_by=[(d1, df1, param1), (d2, df2, param2)]
    :return: List of results
    """
    local_processes = kwargs.pop("local_processes", cpu_count() / 2)

    mpp.Pool.istarmap = istarmap
    with mpp.Pool(processes=round(local_processes)) as pool:
        if all(isinstance(p, tuple) for p in partition_by):
            # Drop into starmap for MP if partition_by is list of tuples
            # frames = pool.starmap(func, tqdm.tqdm(partition_by, total=len(partition_by),
            #                                      desc=kwargs.get('loading_desc')))
            res = tqdm.tqdm(
                pool.istarmap(func, partition_by),
                total=len(partition_by),
                desc=kwargs.get("loading_desc"),
            )
        elif all(isinstance(p, dict) for p in partition_by):
            # Convert a list of dicts into a list of positional tuples
            jobs = [tuple(p.values()) for p in partition_by]
            res = tqdm.tqdm(
                pool.istarmap(func, jobs),
                total=len(partition_by),
                desc=kwargs.get("loading_desc"),
            )
        else:
            it = pool.imap(partial(func, **kwargs), partition_by)
            pbar = tqdm.tqdm(total=len(partition_by))
            res = []
            while True:
                try:
                    res.append(it.next(timeout=60 * 4))
                    pbar.update()
                except mpp.TimeoutError:
                    raise
                except StopIteration:
                    pbar.close()
                    break

        pool.close()
        pool.join()

    return res


def parallel_train_test(
    train_idx,
    test_idx,
    X: pd.DataFrame,
    y: pd.Series,
    model,
    pred_func: str = "predict",
    score_func: str = "rmse",
) -> List[float]:
    """
    Generic train / test function that can be called dynamically in parall from a Jupyter notebook
    :param train_idx: Training data index, to split X, y
    :param test_idx: Testing data index, to split X, y
    :param X: Features to use in the model
    :param y: Target variable
    :param model: The model object instantiated with hyperparamters
    :param pred_func: String value that represents the pred
    """
    assert pred_func in ["predict", "predict_proba"], "Invalid pred_func"

    # Split the data into training and testing sets for each fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    y_pred = getattr(model, pred_func)(X_test)
    # Only need predict_proba for Classification based models
    # y_pred = model_xgb.predict_proba(X_test[features])[:,1]
    print(round(score_func(y_test, y_pred), 4))
    return score_func(y_test, y_pred)
