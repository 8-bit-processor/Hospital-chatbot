qa_pairs = [
    ("I want to schedule an appointment", "ACTION:SCHEDULE_APPOINTMENT"),
    ("Can I book a doctor's visit?", "ACTION:SCHEDULE_APPOINTMENT"),
    ("How do I make an appointment?", "ACTION:SCHEDULE_APPOINTMENT"),

    # Specialty Appointments
    ("I need a neurology appointment", "ACTION:SCHEDULE_NEUROLOGY_APPOINTMENT"),
    ("Schedule an appointment with a neurologist", "ACTION:SCHEDULE_NEUROLOGY_APPOINTMENT"),
    ("I want to see a brain doctor", "ACTION:SCHEDULE_NEUROLOGY_APPOINTMENT"),
    ("Book an orthopedics appointment", "ACTION:SCHEDULE_ORTHOPEDICS_APPOINTMENT"),
    ("I need to see an orthopedic surgeon", "ACTION:SCHEDULE_ORTHOPEDICS_APPOINTMENT"),
    ("My bones hurt, I need an appointment", "ACTION:SCHEDULE_ORTHOPEDICS_APPOINTMENT"),
    ("Dermatology appointment please", "ACTION:SCHEDULE_DERMATOLOGY_APPOINTMENT"),
    ("I have a skin condition, need to see a dermatologist", "ACTION:SCHEDULE_DERMATOLOGY_APPOINTMENT"),
    ("Book a physical therapy session", "ACTION:SCHEDULE_PHYSICAL_THERAPY_APPOINTMENT"),
    ("I need rehab for my injury", "ACTION:SCHEDULE_PHYSICAL_THERAPY_APPOINTMENT"),

    # Medical Imaging
    ("I need a CT scan", "ACTION:MEDICAL_IMAGING_REQUEST"),
    ("Schedule a cat scan", "ACTION:MEDICAL_IMAGING_REQUEST"),
    ("I need an ultrasound", "ACTION:MEDICAL_IMAGING_REQUEST"),
    ("Book an ultrasound appointment", "ACTION:MEDICAL_IMAGING_REQUEST"),
    ("I need an MRI", "ACTION:MEDICAL_IMAGING_REQUEST"),
    ("Schedule an MRI scan", "ACTION:MEDICAL_IMAGING_REQUEST"),

    # Medical Supplies
    ("I need incontinence supplies", "ACTION:MEDICAL_SUPPLIES_REQUEST"),
    ("How can I get adult diapers?", "ACTION:MEDICAL_SUPPLIES_REQUEST"),
    ("I need durable medical equipment", "ACTION:MEDICAL_SUPPLIES_REQUEST"),
    ("How do I get a wheelchair?", "ACTION:MEDICAL_SUPPLIES_REQUEST"),
    ("I need a hospital bed", "ACTION:MEDICAL_SUPPLIES_REQUEST"),

    # Medication Renewal
    ("I need to renew my prescription", "ACTION:RENEW_MEDICATION"),
    ("Can you refill my medication?", "ACTION:RENEW_MEDICATION"),

    # Medical Advice Referral
    ("I need medical advice", "ACTION:MEDICAL_ADVICE_REFERRAL"),
    ("I have a health question", "ACTION:MEDICAL_ADVICE_REFERRAL"),

    # General Information
    ("What are your hospital hours?", "Our hospital is open 24/7 for emergencies. For specific department hours, please visit our website."),
    ("Where is the hospital located?", "We are located at 123 Hospital Road, Anytown, USA."),
    ("What services do you offer?", "We offer a wide range of medical services including emergency care, specialty clinics, surgery, and more. Please visit our website for a full list."),

    # General Conversation
    ("Hello", "Hello! How can I help you today?"),
    ("Hi there", "Hi! What can I do for you?"),
    ("Thank you", "You're welcome!"),
    ("Goodbye", "Goodbye! Have a great day."),
    ("I have a general question", "Please tell me more about your question."),
    ("I need to speak to someone", "Please hold while I connect you to a representative."),
]