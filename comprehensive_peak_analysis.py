import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from itertools import combinations
from sklearn.metrics import r2_score
from scipy import stats

FOLDER = "diverse_datasets"
MIN_SAMPLES = 20

files = [f for f in os.listdir(FOLDER) if f.endswith(".csv")]
files.sort(key=lambda x: int(x.split('_T')[1].split('C')[0]))

# Step 1: Discover all peaks across all datasets
all_detected_peaks = []

for fname in files:
    df = pd.read_csv(os.path.join(FOLDER, fname))
    times = df.iloc[:, 0].values
    intensities = df.iloc[:, 1].values
    
    # Skip if data is invalid
    if len(times) < 10 or len(set(times)) < 5:
        continue
        
    # Adaptive threshold based on data characteristics
    baseline = np.percentile(intensities, 10)
    noise_subset = intensities[intensities < np.percentile(intensities, 20)]
    if len(noise_subset) > 1:
        noise_level = np.std(noise_subset)
    else:
        noise_level = np.std(intensities) * 0.1
    
    threshold = baseline + max(5 * noise_level, 1000)  # Minimum threshold
    
    # Calculate safe distance parameter
    time_step = times[1] - times[0] if len(times) > 1 else 0.01
    if time_step <= 0 or not np.isfinite(time_step):
        time_step = 0.01  # Default 0.01 min step
    
    distance_points = max(1, int(0.2 / time_step))  # At least 1 point separation
    
    # Find peaks with reasonable separation
    peak_indices, properties = find_peaks(
        intensities, 
        height=threshold, 
        distance=distance_points,
        prominence=max(threshold * 0.3, 500)
    )
    
    for idx in peak_indices:
        rt = times[idx]
        height = intensities[idx]
        all_detected_peaks.append((rt, height, fname))

# Step 2: Cluster peaks to find consistent retention times
peak_clusters = []
used_peaks = set()

for i, (rt1, height1, fname1) in enumerate(all_detected_peaks):
    if i in used_peaks:
        continue
    
    cluster = [(rt1, height1, fname1)]
    used_peaks.add(i)
    
    for j, (rt2, height2, fname2) in enumerate(all_detected_peaks):
        if j in used_peaks or j <= i:
            continue
        
        if abs(rt1 - rt2) <= 0.3:  # Cluster peaks within 0.3 min
            cluster.append((rt2, height2, fname2))
            used_peaks.add(j)
    
    if len(cluster) >= MIN_SAMPLES:  # Only keep peaks found in enough samples
        peak_clusters.append(cluster)

# Get representative RT for each cluster
consistent_peaks = []
for cluster in peak_clusters:
    rts = [rt for rt, _, _ in cluster]
    representative_rt = np.median(rts)
    consistent_peaks.append(representative_rt)

consistent_peaks.sort()
print(f"🔍 Peak Discovery: Found {len(consistent_peaks)} consistent peaks at {[f'{rt:.2f}' for rt in consistent_peaks]} min")

# Step 3: Accurate peak integration for all samples
def integrate_peak(times, intensities, target_rt, window=0.75):
    """Integrate peak area around target retention time"""
    if len(times) < 3:
        return 0
        
    # Find closest time point
    center_idx = np.argmin(np.abs(times - target_rt))
    
    # Define integration window with safety checks
    time_step = times[1] - times[0] if len(times) > 1 else 0.01
    if time_step <= 0 or not np.isfinite(time_step):
        time_step = 0.01
    
    window_points = max(3, int(window / time_step))  # At least 3 points
    
    left = max(0, center_idx - window_points)
    right = min(len(times), center_idx + window_points)
    
    # Extract peak region
    peak_times = times[left:right]
    peak_intensities = intensities[left:right]
    
    # Baseline correction (linear baseline between endpoints)
    if len(peak_intensities) > 2 and len(peak_times) == len(peak_intensities):
        baseline_left = peak_intensities[0]
        baseline_right = peak_intensities[-1]
        baseline = np.linspace(baseline_left, baseline_right, len(peak_intensities))
        corrected_intensities = peak_intensities - baseline
        corrected_intensities = np.maximum(corrected_intensities, 0)
        
        # Integrate using trapezoidal rule
        if len(corrected_intensities) > 1 and len(peak_times) > 1:
            area = np.trapz(corrected_intensities, peak_times)
            return max(area, 0)
    
    return 0

# Build complete intensity matrix
temperatures = []
intensity_matrix = []

for fname in files:
    temp = int(fname.split('_T')[1].split('C')[0])
    df = pd.read_csv(os.path.join(FOLDER, fname))
    times = df.iloc[:, 0].values
    intensities = df.iloc[:, 1].values
    
    row = []
    for target_rt in consistent_peaks:
        area = integrate_peak(times, intensities, target_rt)
        row.append(area)
    
    temperatures.append(temp)
    intensity_matrix.append(row)

temperatures = np.array(temperatures)
intensity_matrix = np.array(intensity_matrix)

# Step 4: Enhanced trend differentiation
def linear_model(x, a, b): return a * x + b
def quadratic_model(x, a, b, c): return a * x**2 + b * x + c
def exponential_model(x, a, b, c): return a * np.exp(b * (x - x.min()) / (x.max() - x.min())) + c
def logarithmic_model(x, a, b, c): return a * np.log(x - x.min() + 1) + b * x + c
def power_model(x, a, b, c): return a * np.power(x - x.min() + 1, b) + c

models = {
    'Linear': (linear_model, 2),
    'Quadratic': (quadratic_model, 3),
    'Exponential': (exponential_model, 3),
    'Logarithmic': (logarithmic_model, 3),
    'Power': (power_model, 3)
}

def calculate_aic(y_true, y_pred, n_params):
    n = len(y_true)
    mse = np.mean((y_true - y_pred)**2)
    if mse <= 0: return np.inf
    return n * np.log(mse) + 2 * n_params

def analyze_curvature(x, y):
    """Analyze curvature to distinguish exponential vs logarithmic"""
    if len(x) < 5:
        return 0, 0
    
    # Calculate second derivative using finite differences
    dx = np.diff(x)
    dy = np.diff(y)
    
    if len(dx) < 2:
        return 0, 0
    
    # First derivative
    dy_dx = dy / dx
    
    # Second derivative
    if len(dy_dx) > 1:
        d2y_dx2 = np.diff(dy_dx) / dx[:-1]
        
        # Average curvature
        avg_curvature = np.mean(d2y_dx2)
        
        # Curvature trend (increasing vs decreasing)
        curvature_trend = np.corrcoef(x[1:-1], d2y_dx2)[0,1] if len(d2y_dx2) > 2 else 0
        
        return avg_curvature, curvature_trend
    
    return 0, 0

def test_exponential_vs_logarithmic(x, y, exp_params, log_params):
    """Enhanced test to distinguish exponential from logarithmic"""
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    
    # Generate predictions
    y_exp = exponential_model(x_norm, *exp_params)
    y_log = logarithmic_model(x_norm, *log_params)
    
    # Test 1: Curvature analysis
    avg_curvature, curvature_trend = analyze_curvature(x, y)
    
    # Test 2: Rate of change analysis
    dy_dx = np.gradient(y, x)
    rate_trend = np.corrcoef(x, dy_dx)[0,1] if len(dy_dx) > 2 else 0
    
    # Test 3: End-point behavior
    if len(y) >= 4:
        early_change = abs(y[len(y)//4] - y[0]) / (x[len(x)//4] - x[0]) if x[len(x)//4] != x[0] else 0
        late_change = abs(y[-1] - y[3*len(y)//4]) / (x[-1] - x[3*len(x)//4]) if x[-1] != x[3*len(x)//4] else 0
        change_acceleration = late_change / early_change if early_change > 0 else 1
    else:
        change_acceleration = 1
    
    # Scoring system
    exp_score = 0
    log_score = 0
    
    # Curvature scoring
    if avg_curvature > 0.01:
        exp_score += 2
    elif avg_curvature < -0.01:
        log_score += 2
    
    # Rate trend scoring
    if rate_trend > 0.3:
        exp_score += 3
    elif rate_trend < -0.3:
        log_score += 3
    
    # Change acceleration scoring
    if change_acceleration > 1.5:
        exp_score += 2
    elif change_acceleration < 0.7:
        log_score += 2
    
    return exp_score, log_score, {
        'avg_curvature': avg_curvature,
        'rate_trend': rate_trend,
        'change_acceleration': change_acceleration
    }

def characterize_trend(x, y, model_name, params, diagnostics=None):
    if model_name == 'Linear':
        slope = params[0]
        return 'Increasing' if slope > 0 else 'Decreasing'
    elif model_name == 'Quadratic':
        return 'Quadratic'
    elif model_name == 'Exponential':
        base_type = 'Exponential Growth' if params[1] > 0 else 'Exponential Decay'
        if diagnostics and 'rate_trend' in diagnostics:
            if diagnostics['rate_trend'] > 0.5:
                return f'{base_type} (Strong)'
            elif diagnostics['change_acceleration'] > 2.0:
                return f'{base_type} (Accelerating)'
        return base_type
    elif model_name == 'Logarithmic':
        base_type = 'Logarithmic'
        if diagnostics and 'rate_trend' in diagnostics:
            if diagnostics['rate_trend'] < -0.5:
                return 'Logarithmic (Strong)'
            elif diagnostics['change_acceleration'] < 0.5:
                return 'Logarithmic (Saturating)'
        return base_type
    elif model_name == 'Power':
        exp = params[1]
        return 'Accelerating' if exp > 1 else 'Decelerating' if 0 < exp < 1 else 'Power'
    return 'Unknown'

def fit_model_with_selection(x, y):
    results = []
    
    for model_name, (model_func, n_params) in models.items():
        try:
            x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
            
            if model_name == 'Exponential':
                y_range = y.max() - y.min()
                p0 = [y_range, 1.0, y.min()]
                bounds = ([-5*y_range, -10, y.min()-y_range], [5*y_range, 10, y.max()+y_range])
                popt, _ = curve_fit(model_func, x_norm, y, p0=p0, bounds=bounds, maxfev=10000)
            elif model_name == 'Power':
                y_range = y.max() - y.min()
                p0 = [y_range, 1.0, y.min()]
                bounds = ([0.1*y_range, 0.1, y.min()-y_range], [10*y_range, 5, y.max()+y_range])
                popt, _ = curve_fit(model_func, x_norm, y, p0=p0, bounds=bounds, maxfev=10000)
            else:
                popt, _ = curve_fit(model_func, x_norm, y, maxfev=10000)
            
            y_pred = model_func(x_norm, *popt)
            
            # Remove outliers
            residuals = np.abs(y - y_pred)
            Q3 = np.percentile(residuals, 75)
            IQR = np.percentile(residuals, 75) - np.percentile(residuals, 25)
            threshold = Q3 + 1.5 * IQR
            mask = residuals <= threshold
            
            if mask.sum() < MIN_SAMPLES:
                mask = np.ones(len(x), dtype=bool)
            
            # Refit with outliers removed
            if model_name == 'Exponential':
                popt_clean, _ = curve_fit(model_func, x_norm[mask], y[mask], p0=p0, bounds=bounds, maxfev=10000)
            elif model_name == 'Power':
                popt_clean, _ = curve_fit(model_func, x_norm[mask], y[mask], p0=p0, bounds=bounds, maxfev=10000)
            else:
                popt_clean, _ = curve_fit(model_func, x_norm[mask], y[mask], maxfev=10000)
            
            y_pred_clean = model_func(x_norm[mask], *popt_clean)
            r2 = r2_score(y[mask], y_pred_clean)
            aic = calculate_aic(y[mask], y_pred_clean, n_params)
            
            results.append({
                'model': model_name,
                'r2': r2,
                'aic': aic,
                'params': popt_clean,
                'mask': mask,
                'n_samples': mask.sum()
            })
        except:
            continue
    
    if not results:
        return None
    
    # Enhanced exponential vs logarithmic comparison
    exp_result = next((r for r in results if r['model'] == 'Exponential'), None)
    log_result = next((r for r in results if r['model'] == 'Logarithmic'), None)
    
    if exp_result and log_result and abs(exp_result['r2'] - log_result['r2']) < 0.05:
        # Close R² values - use enhanced discrimination
        exp_score, log_score, diagnostics = test_exponential_vs_logarithmic(
            x, y, exp_result['params'], log_result['params']
        )
        
        # Adjust scores based on discrimination test
        if exp_score > log_score + 2:
            exp_result['score_boost'] = 0.02
            log_result['score_boost'] = -0.02
        elif log_score > exp_score + 2:
            log_result['score_boost'] = 0.02
            exp_result['score_boost'] = -0.02
        else:
            exp_result['score_boost'] = 0
            log_result['score_boost'] = 0
        
        # Store diagnostics for trend characterization
        exp_result['diagnostics'] = diagnostics
        log_result['diagnostics'] = diagnostics
    
    # Add trend characterization with diagnostics
    for result in results:
        diagnostics = result.get('diagnostics', None)
        result['trend_type'] = characterize_trend(x, y, result['model'], result['params'], diagnostics)
    
    # Select best model using enhanced scoring
    for result in results:
        if result['r2'] > 0.3:
            base_score = result['r2'] - (result['aic'] - min(r['aic'] for r in results if r['r2'] > 0.3)) / 1000
            boost = result.get('score_boost', 0)
            result['score'] = base_score + boost
        else:
            result['score'] = 0
    
    best_result = max(results, key=lambda x: x['score'])
    return best_result if best_result['r2'] > 0.5 else None

best_relationships = []

for i, j in combinations(range(len(consistent_peaks)), 2):
    peak1_intensities = intensity_matrix[:, i]
    peak2_intensities = intensity_matrix[:, j]
    
    # Only use samples where both peaks are detected
    valid_mask = (peak1_intensities > 0) & (peak2_intensities > 0)
    
    if valid_mask.sum() < MIN_SAMPLES:
        continue
    
    ratios = peak1_intensities[valid_mask] / peak2_intensities[valid_mask]
    temps = temperatures[valid_mask]
    
    # Remove extreme ratio outliers
    Q1, Q3 = np.percentile(ratios, [25, 75])
    IQR = Q3 - Q1
    ratio_mask = (ratios >= Q1 - 2*IQR) & (ratios <= Q3 + 2*IQR)
    
    if ratio_mask.sum() < MIN_SAMPLES:
        continue
    
    ratios = ratios[ratio_mask]
    temps = temps[ratio_mask]
    
    # Enhanced model selection
    result = fit_model_with_selection(temps, ratios)
    
    if result and result['r2'] > 0.5:
        rt1, rt2 = consistent_peaks[i], consistent_peaks[j]
        
        # Debug: Show ratio range and trend
        ratio_range = ratios[result['mask']].max() - ratios[result['mask']].min()
        temp_range = temps[result['mask']].max() - temps[result['mask']].min()
        
        best_relationships.append({
            'peak1_rt': rt1,
            'peak2_rt': rt2,
            'model': result['model'],
            'r2': result['r2'],
            'aic': result['aic'],
            'params': result['params'],
            'ratios': ratios[result['mask']],
            'temps': temps[result['mask']],
            'n_samples': result['n_samples'],
            'trend_type': result['trend_type'],
            'ratio_range': ratio_range,
            'temp_range': temp_range
        })

# Sort by R² and display results
best_relationships.sort(key=lambda x: x['r2'], reverse=True)

print(f"\n🎯 Temperature-Ratio Relationships:")
print(f"{'Peak Pair':<12} {'Model':<12} {'Trend Type':<18} {'R²':<6} {'AIC':<8} {'Samples':<8}")
print("-" * 75)

for rel in best_relationships[:10]:
    rt1, rt2 = rel['peak1_rt'], rel['peak2_rt']
    print(f"RT{rt1:.1f}/RT{rt2:.1f}   {rel['model']:<12} {rel['trend_type']:<18} {rel['r2']:.3f}  {rel['aic']:<8.1f} {rel['n_samples']:<8}")

if best_relationships:
    # Enhanced visualization
    best = best_relationships[0]
    
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.scatter(best['temps'], best['ratios'], alpha=0.6, s=50, color='blue')
    
    # Plot fitted curve
    temp_range = np.linspace(best['temps'].min(), best['temps'].max(), 100)
    temp_norm = (temp_range - best['temps'].min()) / (best['temps'].max() - best['temps'].min()) if best['temps'].max() > best['temps'].min() else temp_range
    model_func = models[best['model']][0]
    fitted_curve = model_func(temp_norm, *best['params'])
    plt.plot(temp_range, fitted_curve, 'r-', linewidth=2, 
             label=f"{best['model']} - {best['trend_type']}\nR² = {best['r2']:.3f}, AIC = {best['aic']:.1f}")
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel(f"Peak Ratio (RT{best['peak1_rt']:.1f}/RT{best['peak2_rt']:.1f})")
    plt.title('Best Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show top 3 different trend types
    unique_trends = []
    for rel in best_relationships:
        if rel['trend_type'] not in [u['trend_type'] for u in unique_trends] and len(unique_trends) < 3:
            unique_trends.append(rel)
    
    for idx, rel in enumerate(unique_trends[1:], 2):
        if idx > 4: break
        plt.subplot(2, 2, idx)
        plt.scatter(rel['temps'], rel['ratios'], alpha=0.6, s=30)
        
        temp_range = np.linspace(rel['temps'].min(), rel['temps'].max(), 100)
        temp_norm = (temp_range - rel['temps'].min()) / (rel['temps'].max() - rel['temps'].min()) if rel['temps'].max() > rel['temps'].min() else temp_range
        model_func = models[rel['model']][0]
        fitted_curve = model_func(temp_norm, *rel['params'])
        plt.plot(temp_range, fitted_curve, 'r-', linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel(f"RT{rel['peak1_rt']:.1f}/RT{rel['peak2_rt']:.1f}")
        plt.title(f"{rel['trend_type']} (R² = {rel['r2']:.3f})")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("No strong relationships found (R² > 0.5)")