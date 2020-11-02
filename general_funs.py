from __future__ import division
import numpy as np

def canonicalisation(data, item_dim, legal_range = None):
    """
    Parameters
    ----------
    data: Any numeric array, consisting of single or multiple items.
    item_dim: The canonical dimensionality of each item in the data.
    legal_range: If any individual value is outside legal_range, the data cannot be processed.
    
    Returns
    -------
    y: A n-D numpy array, whose 0th axis contains the item index,
       and the dimensionality of each item is item_dim.
    """
    x = np.asarray(data).flatten()
    # check individual data
    if legal_range is not None:
        if isinstance(legal_range, list):
            for i in x:
                if i not in legal_range:
                    raise ValueError("The data contains at least one illegal value:", i)
        elif isinstance(legal_range, tuple):
            for i in x:
                if i < legal_range[0] or i > legal_range[1]:
                    raise ValueError("The data contains at least one illegal value:", i)
    # obtain the canonical shape of data
    y = x.reshape(np.insert(item_dim, 0, x.size / np.prod(item_dim)))
    return y

def collapse(probs):
    """
    Parameters
    ----------
    probs: An array of probablities.
    
    Returns
    -------
    states: An array of binary states, whose dimensionality is identical to probs.
    """
    probs = np.asarray(probs)
    states = probs >= np.random.rand(*probs.shape)
    return states * 1

def logistic(x, temperature = 1.0):
    """
    Parameters
    ----------
    x: A numeric array.
    temperature: A non-negative number controlling the slope of the function.
    
    Returns
    -------
    y: The value of the function, which is often used as a probability.
    
    -------
    The function is numerically stable for very big/small values.
    """
    if temperature == 0:
        # The logistic function is reduced to a step function.
        y = np.zeros(x.shape)
        y[x > 0] = 1.0
        y[x == 0] = 0.5     
    else:
        norx = np.asarray(x) / temperature
        mask_p = norx >= 0
        mask_n = norx < 0        
        y = np.ones_like(norx)
        y[mask_p] = 1 / (1 + np.exp(-norx[mask_p]))
        # positive x gives small exp(-x): 1<denom<2
        z = np.zeros_like(y[mask_n])
        z = np.exp(norx[mask_n])
        y[mask_n] = z / (1 + z)        
        # negative x gives small exp(x)=z: 1<denom<2
    return y

def softmax(x, temperature = 1.0):
    """
    Parameters
    ----------
    x: A two-dimensional numeric array.
    temperature: A non-negative number controlling the slope of the function.
    
    Returns
    -------
    y: The value of the function, which is often used as a probability. Each row adds up to 1.
    
    -------
    The function is numerically stable for very big/small values.
    """
    x = np.asarray(x)
    item_size = x.shape[-1]
    canx = canonicalisation(x, item_size)
    probs = []
    if temperature == 0:
        for item in canx:
            prob = np.zeros(item_size)
            maxids = np.argwhere(item == np.amax(item))
            prob[maxids] = 1 / len(maxids)
            probs.append(prob)
    else:
        for item in canx:
            noritem = (item - np.amax(item)) / temperature
            pitem = np.exp(noritem)
            prob = pitem / np.sum(pitem)
            probs.append(prob)            
             
    y = canonicalisation(probs, item_size)
    return y

def entropy(P, base = None):
    """
    Parameters
    ----------
    P: A discrete probability distribution listed in a numeric array.
    base: The logarithmic base when calculating entropy with the default value being e.
    
    Returns
    -------
    H: The entropy of the distribution P.
    """
    ps = P / np.sum(P)
    ps[ps == 0] = 1 # plog(p) = when p = 0 or 1
    if base is None:
        denom = 1
    else:
        denom = np.log(base)
    logps = np.log(ps) / denom
    return -np.dot(ps, logps)
    