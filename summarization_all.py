import os
import pandas as pd
import openai
import re
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def summarize_text(text, model="gpt-4o", max_tokens=300):
    prompt = f""""""
    

    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un assistant expert en synthèse de contenus de formation en français."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Tentative {attempt + 1} échouée pour la synthèse : {e}")
    return ""

def generate_content_for_single_course(cluster_name, course_title="", model="gpt-4o", max_tokens=500):
    """Generate content for clusters with only one course"""
    prompt = f""" """

    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un expert en création de contenu pour des formations professionnelles en français."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Tentative {attempt + 1} échouée pour la génération de contenu : {e}")
    return ""

def format_price(price):
    try:
        return f"{int(round(price)):,} € HT".replace(",", " ")
    except:
        return "0 € HT"

def format_duration(days):
    try:
        return f"{int(round(days))} jours"
    except:
        return "0 jours"

def extract_organization_from_url(url):
    if not isinstance(url, str):
        return ""
    url = url.lower()
    if "://" in url:
        url = url.split("://")[1]
    parts = url.split(".")
    if parts[0] == "www" and len(parts) > 1:
        return parts[1]
    return parts[0]

def parse_price(price_str):
    if not isinstance(price_str, str):
        return None

    try:
        # Clean the string
        cleaned = price_str.strip().replace("€", "").replace("HT", "")
        cleaned = re.sub(r"[\u202f\u00a0\s]", "", cleaned)  # remove non-breaking spaces & whitespaces

        # Extract the first number pattern: supports both comma and dot
        match = re.search(r"(\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{2})|\d+(?:[\.,]\d{2}))", cleaned)
        if not match:
            return None

        number_str = match.group(1)

        # Determine if it's French (comma decimal) or English (dot decimal)
        if "," in number_str and not "." in number_str:
            # French format: "975,00"
            number_str = number_str.replace(",", ".")
        elif "," in number_str and "." in number_str:
            # English format with comma as thousands separator: "1,275.00"
            number_str = number_str.replace(",", "")

        return float(number_str)

    except Exception as e:
        logging.warning(f"Erreur de parsing du prix : '{price_str}' → {e}")
        return None

def parse_duration(duration_str):
    if not isinstance(duration_str, str):
        return None

    try:
        duration_str = duration_str.lower()

        # Extract all numbers with units
        hours_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:h|heures?)', duration_str)
        days_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:j|jours?)', duration_str)

        hours = float(hours_match.group(1).replace(',', '.')) if hours_match else 0
        days = float(days_match.group(1).replace(',', '.')) if days_match else 0

        # Handle special cases first
        if 'sur' in duration_str and hours and days:
            # "35 heures sur 5 jours" - use hours as the actual training time
            return round(hours / 7, 2)
        elif days and hours and ('-' in duration_str or 'soit' in duration_str or '(' in duration_str):
            # "5j - 35h00" or "5 jours soit 35h" or "3 jours (21 heures)" - equivalent descriptions
            # Use the days value as it's more reliable for training duration
            return round(days, 2)
        elif days and hours:
            # If both exist without clear relationship, prefer days but validate with hours
            expected_hours = days * 7  # 7 hours per day standard
            if abs(hours - expected_hours) <= 2:  # Allow small variance
                return round(days, 2)
            else:
                # Hours and days don't match standard ratio, use hours
                return round(hours / 7, 2)
        elif days:
            return round(days, 2)
        elif hours:
            return round(hours / 7, 2)
        else:
            return None

    except Exception as e:
        logging.warning(f"Erreur de parsing durée '{duration_str}' → {e}")
        return None

def clean_text_for_storage(text):
    """
    Clean text data to remove problematic whitespace while maintaining content
    """
    if not isinstance(text, str):
        return "N/A"
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text.strip())
    return text if text else "N/A"

def extract_competitor_price(competitor_str):
    """
    Extracts price from competitor string in the format:
    organization--->description--(price),(price),(price),(url),(duration),(level)
    
    Returns the first valid price found or None.
    """
    if not isinstance(competitor_str, str):
        return None
        
    try:
        # Match price patterns in parentheses
        price_matches = re.findall(r'\(([^)]*)\)', competitor_str)
        
        # Check up to the first 3 matches which should be prices
        for i in range(min(3, len(price_matches))):
            price = parse_price(price_matches[i])
            if price and price > 0:
                return price
                
        return None
    except Exception as e:
        logging.warning(f"Erreur d'extraction du prix concurrent : {e}")
        return None

def get_default_failback_price(duration):
    """
    Returns default failback prices based on duration as specified by client.
    Handles fractional durations and extrapolation for longer courses.
    """
    # Round duration to handle fractional days
    duration_rounded = round(duration, 1)
    
    # Exact matches for standard durations
    default_prices = {
        1: 980,
        2: 1950, 
        3: 2930,
        4: 3910,
        5: 4875
    }
    
    # Check for exact match first
    duration_int = int(duration_rounded) if duration_rounded == int(duration_rounded) else None
    if duration_int and duration_int in default_prices:
        return default_prices[duration_int]
    
    # Handle fractional durations (interpolate)
    if duration_rounded < 1:
        return int(980 * duration_rounded)
    elif duration_rounded < 5:
        # Linear interpolation between known points
        lower_day = int(duration_rounded)
        upper_day = lower_day + 1
        
        if lower_day in default_prices and upper_day in default_prices:
            lower_price = default_prices[lower_day]
            upper_price = default_prices[upper_day]
            fraction = duration_rounded - lower_day
            interpolated_price = lower_price + (upper_price - lower_price) * fraction
            return int(interpolated_price)
    
    # Handle durations > 5 days (extrapolate)
    if duration_rounded > 5:
        # Base price for 5 days + additional days at 975€/day rate
        base_price = default_prices[5]  # 4875€
        additional_days = duration_rounded - 5
        additional_cost = additional_days * 975
        return int(base_price + additional_cost)
    
    # Fallback (shouldn't reach here, but safety net)
    return int(980 * duration_rounded)

def calculate_internal_selling_price(cluster_df, actual_duration):   
    # Extract organization from URL for identification
    cluster_df['organization_from_url'] = cluster_df['url'].apply(extract_organization_from_url)
    
    # PRIORITY 1: Check for DAWAN (highest priority)
    dawan_rows = cluster_df[cluster_df['organization_from_url'].str.contains("dawan", na=False, case=False)]
    
    if not dawan_rows.empty:
        logging.info(" - applying FIXED Dawan pricing logic")
        
        # Try Dawan's intra-company prices first (preferred)
        dawan_intra_prices = dawan_rows['intra_company_price'].dropna()
        
        # Parse prices - FIXED: These are already daily rates!
        parsed_intra_prices = []
        
        for price_str in dawan_intra_prices:
            parsed_price = parse_price(str(price_str))
            if parsed_price and parsed_price > 0:
                parsed_intra_prices.append(parsed_price)
        
        if parsed_intra_prices:
            # FIXED: Use the daily rate directly (don't divide by duration)
            if len(parsed_intra_prices) == 1:
                dawan_daily_rate = parsed_intra_prices[0]
            else:
                # Average multiple daily rates
                dawan_daily_rate = sum(parsed_intra_prices) / len(parsed_intra_prices)
                logging.info(f"Multiple intra prices found, using average daily rate: {dawan_daily_rate:.2f}€")
            
            if dawan_daily_rate > 50:
                # Apply -50€ discount per day and multiply by actual duration
                adjusted_daily_rate = dawan_daily_rate - 50
                internal_selling_price = adjusted_daily_rate * actual_duration
                logging.info(f"FIXED pricing applied (intra): ({dawan_daily_rate:.2f} - 50) × {actual_duration} = {internal_selling_price:.2f}€")
                return internal_selling_price, "dawan_intra"
        
        # Fallback to Dawan individual prices if no valid intra prices
        logging.info("")
        dawan_indiv_prices = dawan_rows['individual_price'].dropna()
        dawan_durations = dawan_rows['duration'].dropna()
        
        parsed_indiv_prices = []
        parsed_durations = []
        
        for price_str in dawan_indiv_prices:
            parsed_price = parse_price(str(price_str))
            if parsed_price and parsed_price > 0:
                parsed_indiv_prices.append(parsed_price)
        
        for duration_str in dawan_durations:
            parsed_duration = parse_duration(str(duration_str))
            if parsed_duration and parsed_duration > 0:
                parsed_durations.append(parsed_duration)
        
        if parsed_indiv_prices and parsed_durations:
            # Calculate daily rate from individual prices (these are NOT daily rates)
            if len(parsed_indiv_prices) == 1 and len(parsed_durations) >= 1:
                dawan_daily_rate = parsed_indiv_prices[0] / parsed_durations[0]
            else:
                daily_rates = []
                for i in range(min(len(parsed_indiv_prices), len(parsed_durations))):
                    if parsed_durations[i] > 0:
                        daily_rates.append(parsed_indiv_prices[i] / parsed_durations[i])
                
                if daily_rates:
                    dawan_daily_rate = sum(daily_rates) / len(daily_rates)
                else:
                    dawan_daily_rate = None
            
            if dawan_daily_rate and dawan_daily_rate > 50:
                # Apply -50€ discount per day and multiply by actual duration
                adjusted_daily_rate = dawan_daily_rate - 50
                internal_selling_price = adjusted_daily_rate * actual_duration
                logging.info(f"pricing applied (individual fallback): ({dawan_daily_rate:.2f} - 50) × {actual_duration} = {internal_selling_price:.2f}€")
                return internal_selling_price, ""
        
        logging.warning("ound but no valid pricing data available, falling back to next priority")
    
    # PRIORITY 2: Check for CEGOS (only if no failed)
    cegos_rows = cluster_df[cluster_df['organization_from_url'].str.contains("", na=False, case=False)]
    
    if not cegos_rows.empty:
        logging.info(" - applying pricing logic")
        
        # Extract Cegos inter-company prices
        cegos_inter_prices = cegos_rows['inter_company_price'].dropna()
        
        parsed_cegos_prices = []
        for price_str in cegos_inter_prices:
            parsed_price = parse_price(str(price_str))
            if parsed_price and parsed_price > 0:
                parsed_cegos_prices.append(parsed_price)
        
        if parsed_cegos_prices:
            # Use first valid price or average if multiple
            if len(parsed_cegos_prices) == 1:
                cegos_inter_price = parsed_cegos_prices[0]
            else:
                cegos_inter_price = sum(parsed_cegos_prices) / len(parsed_cegos_prices)
                logging.info(f"Multiple prices found, using average: {cegos_inter_price:.2f}€")
            
            internal_selling_price = cegos_inter_price * 3
            logging.info(f"pricing applied: {cegos_inter_price:.2f} × 3 = {internal_selling_price:.2f}€")
            return internal_selling_price, ""
        else:
            logging.warning("found but no valid inter-company prices, falling back to default")
    
    # PRIORITY 3: Default failback pricing table
    internal_selling_price = get_default_failback_price(actual_duration)
    logging.info(f"Default failback pricing applied for {actual_duration} days: {internal_selling_price}€")
    return internal_selling_price, "default_failback"

def generate_cluster_files(input_file: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(input_file, engine='openpyxl')

    # Check if Cluster column exists
    if 'Cluster' not in df.columns:
        logging.error("Colonne 'Cluster' non trouvée dans le fichier d'entrée")
        return

    text_fields = ['context', 'target_audience', 'objectives', 'programme', 'prerequisites', 'pedagogy', 'highlights', 'financing_options']

    # Process all clusters in the file
    clusters = df['Cluster'].dropna().unique()

    logging.info(f"Starting processing for {len(clusters)} clusters found in the file")

    if len(clusters) == 0:
        logging.warning("No clusters found in the input file")
        return
        
    # Count clusters by size for logging
    small_clusters = 0
    medium_clusters = 0
    large_clusters = 0
    for cluster in clusters:
        course_count = len(df[df['Cluster'] == cluster])
        if course_count == 1:
            small_clusters += 1
        elif 2 <= course_count <= 5:
            medium_clusters += 1
        else:
            large_clusters += 1
            
    logging.info(f"Cluster distribution: {small_clusters} single courses, {medium_clusters} medium clusters (2-5 courses), {large_clusters} large clusters (>5 courses)")

    # Process each cluster
    processed_count = 0
    skipped_count = 0

    for idx, cluster in enumerate(clusters):
        cluster_df = df[df['Cluster'] == cluster].copy()
        course_count = len(cluster_df)
        
        cluster_name_safe = re.sub(r'[\\/*?:"<>|]', "-", cluster)
        out_path = os.path.join(output_folder, f"{cluster_name_safe}.xlsx")
        
        # Only check if the file exists in the output folder
        if os.path.exists(out_path):
            logging.info(f"Cluster {cluster} ({idx+1}/{len(clusters)}): Fichier déjà existant dans le dossier de sortie (ignoré)")
            skipped_count += 1
            continue

        logging.info(f"Traitement du cluster: {cluster} ({idx+1}/{len(clusters)}, {course_count} formations)")
        summary_record = {'Cluster': cluster}

        # Handle different processing based on number of courses
        if course_count >= 2:
            # For 2+ courses, summarize content
            if 'title' in cluster_df.columns:
                titles_combined = ' '.join(cluster_df['title'].dropna().astype(str).tolist())
                summary_record['title'] = summarize_text(titles_combined) if titles_combined.strip() else ""
            else:
                summary_record['title'] = ""

            for field in text_fields:
                if field not in cluster_df.columns:
                    summary_record[field] = ""
                    continue

                combined_text = ' '.join(cluster_df[field].dropna().astype(str).tolist())
                summary_record[field] = summarize_text(combined_text) if combined_text.strip() else ""
                
        else:  # For single course
            # Extract course title if available
            course_title = cluster_df['title'].iloc[0] if 'title' in cluster_df.columns and not cluster_df['title'].empty else ""
            
            # Generate thematic summary for single course
            theme_summary = generate_content_for_single_course(cluster, course_title)
            
            # Set the generated theme as the main context
            summary_record['context'] = theme_summary
            
            # For other fields, use original content if available or leave blank
            if 'title' in cluster_df.columns and not cluster_df['title'].empty:
                summary_record['title'] = cluster_df['title'].iloc[0]
            else:
                summary_record['title'] = cluster
                
            # Copy other fields directly if available
            for field in text_fields:
                if field == 'context':  # Already handled above
                    continue
                    
                if field in cluster_df.columns and not cluster_df[field].empty:
                    summary_record[field] = cluster_df[field].iloc[0]
                else:
                    summary_record[field] = ""

        try:
            # PRICING CALCULATION
            
            # Fixed values for cost calculations
            trainer_day_rate = 400  
            
            # Calculate average duration from all courses in cluster
            all_durations = cluster_df['duration'].dropna().astype(str).apply(parse_duration).dropna()
            avg_duration = all_durations.mean() if not all_durations.empty else 2
            actual_duration = round(avg_duration)
            
            # Calculate individual price using FIXED pricing logic
            individual_price, individual_pricing_source = calculate_internal_selling_price(cluster_df, actual_duration)
            
            # Calculate internal selling price using same FIXED logic as individual price
            internal_selling_price, internal_pricing_source = calculate_internal_selling_price(cluster_df, actual_duration)
            
            # Calculate cost price (Prix d'achat HT)
            cost_price = trainer_day_rate * actual_duration
            
            # Store pricing information
            summary_record['Durée (jours)'] = format_duration(actual_duration)
            summary_record['Prix individuel HT'] = format_price(individual_price)
            summary_record['Prix d\'achat HT'] = format_price(cost_price)
            summary_record['Prix de vente interne HT'] = format_price(internal_selling_price)
            
            # Add pricing source for debugging
            summary_record['individual_pricing_source'] = individual_pricing_source
            summary_record['internal_pricing_source'] = internal_pricing_source
            
            logging.info(f"Cluster {cluster}: Duration={actual_duration} days, Individual Price={individual_price:.2f}€ (source: {individual_pricing_source}), Cost Price={cost_price}€, Internal Selling Price={internal_selling_price:.2f}€ (source: {internal_pricing_source})")

        except Exception as e:
            logging.error(f"Cluster {cluster}: Erreur de calcul des prix - {str(e)}")
            summary_record['Durée (jours)'] = "0 jours"
            summary_record['Prix individuel HT'] = "0 € HT"
            summary_record['Prix d\'achat HT'] = "0 € HT"
            summary_record['Prix de vente interne HT'] = "0 € HT"
            summary_record['individual_pricing_source'] = "error"
            summary_record['internal_pricing_source'] = "error"

        # Handle URL field
        if 'url' in cluster_df.columns:
            urls = cluster_df['url'].dropna().astype(str).tolist()
            summary_record['url'] = '\n'.join(urls)
        else:
            summary_record['url'] = ""

        # Create competitors_prices field
        try:
            competitors_list = []
            for _, row in cluster_df.iterrows():
                # Extract organization name from URL
                organization = extract_organization_from_url(row.get('url', ''))
                if not organization:
                    continue  # Skip entries without organization
                
                # Clean text fields to remove problematic whitespace while preserving content
                title = clean_text_for_storage(row.get('title', ''))
                indiv_price = clean_text_for_storage(row.get('individual_price', ''))
                inter_price = clean_text_for_storage(row.get('inter_company_price', ''))
                intra_price = clean_text_for_storage(row.get('intra_company_price', ''))
                url = clean_text_for_storage(row.get('url', ''))
                duration = clean_text_for_storage(row.get('duration', ''))
                level = clean_text_for_storage(row.get('level', ''))
                
                # Format in the original style but with cleaner data
                competitor_entry = (
                    f"{organization}--->{title}--"
                    f"({indiv_price}),({inter_price}),({intra_price}),"
                    f"({url}),({duration}),({level})"
                )
                
                competitors_list.append(competitor_entry)
            
            # Deduplicate entries
            competitors_list = list(dict.fromkeys(competitors_list))
            
            # Join with the same ' | ' delimiter as original
            summary_record['competitors_prices'] = ' | '.join(competitors_list)

        except Exception as e:
            logging.error(f"Cluster {cluster}: Erreur lors de la création de competitors_prices - {str(e)}")
            summary_record['competitors_prices'] = ""

        # Save to Excel file
        pd.DataFrame([summary_record]).to_excel(out_path, index=False)
        logging.info(f"Cluster {cluster}: Fichier créé avec succès → {out_path}")
        processed_count += 1
        
    logging.info(f"Résumé du traitement: {processed_count} clusters traités, {skipped_count} clusters ignorés")

if __name__ == "__main__":
    input_file = ""  # Modify this if needed
    output_folder = ""

    logging.info(f"Démarrage du traitement pour tous les clusters")

    # Add debug info about the input file
    if os.path.exists(input_file):  
        try:
            df = pd.read_excel(input_file, engine='openpyxl')
            total_rows = len(df)
            unique_clusters = df['Cluster'].dropna().nunique() if 'Cluster' in df.columns else 0
            logging.info(f"Le fichier d'entrée contient {total_rows} lignes et {unique_clusters} clusters uniques")
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier d'entrée pour debug: {e}")
    else:
        logging.error(f"Le fichier d'entrée '{input_file}' n'existe pas")

    generate_cluster_files(input_file, output_folder)
    logging.info("Traitement terminé")


