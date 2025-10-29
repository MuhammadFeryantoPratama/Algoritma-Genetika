# Simple fuzzy controller (temperature -> fan speed)
def mf_low(x):
    if x <= 20: return 1.0
    if 20 < x < 25: return (25 - x) / 5.0
    return 0.0

def mf_medium(x):
    if 20 < x < 25: return (x - 20) / 5.0
    if 25 <= x <= 30: return (30 - x) / 5.0
    return 0.0

def mf_high(x):
    if x <= 28: return 0.0
    if 28 < x < 33: return (x - 28) / 5.0
    return 1.0 if x >= 33 else 0.0

def defuzzify(activations):
    samples = 201
    xs = [i * (100/(samples-1)) for i in range(samples)]
    num = 0.0
    den = 0.0
    for x in xs:
        # output membership approximations
        slow = 1.0 if x <= 20 else max(0.0, (40 - x) / 20.0) if x <= 40 else 0.0
        med = 0.0
        if 30 < x < 70:
            med = min((x-30)/40.0, (70-x)/40.0)
        elif 30 <= x <= 70:
            med = 1.0
        fast = 1.0 if x >= 80 else max(0.0, (x - 60) / 20.0) if x >= 60 else 0.0

        val = 0.0
        if activations.get('slow', 0) > 0:
            val = max(val, min(slow, activations['slow']))
        if activations.get('medium', 0) > 0:
            val = max(val, min(med, activations['medium']))
        if activations.get('fast', 0) > 0:
            val = max(val, min(fast, activations['fast']))

        num += val * x
        den += val
    return (num/den) if den > 0 else 0.0

def evaluate_temperature(temp_celsius):
    a_low = mf_low(temp_celsius)
    a_med = mf_medium(temp_celsius)
    a_high = mf_high(temp_celsius)
    activations = {'slow': a_low, 'medium': a_med, 'fast': a_high}
    fan_speed = defuzzify(activations)
    if fan_speed < 33:
        label = 'Pelan'
    elif fan_speed < 66:
        label = 'Sedang'
    else:
        label = 'Cepat'
    return {
        'temperature': temp_celsius,
        'activations': activations,
        'fan_speed': round(fan_speed, 2),
        'label': label
    }
