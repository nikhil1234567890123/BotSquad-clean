# intent_classifier.py

def classify_intent(query: str) -> str:
    query = query.lower()

    if any(word in query for word in ["admission", "entrance", "apply"]):
        return "admission"
    
    elif any(word in query for word in ["exam", "exams", "datesheet", "exam date", "result", "results", "marksheet"]):
        return "exam"
    
    elif any(word in query for word in ["fee", "payment", "fees", "pay online"]):
        return "fee"
    
    elif any(word in query for word in ["faculty", "professor", "teacher", "staff"]):
        return "faculty"
    
    elif any(word in query for word in ["hostel", "hostels", "accommodation", "residence"]):
        return "hostel"
    
    elif any(word in query for word in ["placement", "job", "career", "internship", "placements", "jobs", "careers", "internships"]):
        return "placement"
    
    elif any(word in query for word in ["syllabus", "curriculum", "subjects", "subject"]):
        return "syllabus"
    
    else:
        return "unknown"
