from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Response
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlmodel import SQLModel, Field, Relationship, create_engine, Session, select
import os
from ics import Calendar, Event
from math import radians, cos, sin, asin, sqrt

# ------------------------------
# DB & app
# ------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///./crm.db")
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)

def get_session():
    with Session(engine) as s:
        yield s

app = FastAPI(title="CRM Tournées — MVP", version="0.1.0")

# ------------------------------
# Models
# ------------------------------
class ClientBase(SQLModel):
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = "France"
    lat: Optional[float] = None
    lon: Optional[float] = None
    visit_frequency_days: Optional[int] = None  # ex: 30
    priority: Optional[int] = 0
    notes: Optional[str] = None

class Client(ClientBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appointments: List["Appointment"] = Relationship(back_populates="client")
    followups: List["FollowUp"] = Relationship(back_populates="client")

class ClientCreate(ClientBase): pass
class ClientRead(ClientBase):
    id: int

class AppointmentBase(SQLModel):
    client_id: int = Field(foreign_key="client.id")
    start: datetime
    end: datetime
    subject: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    is_planned: bool = True
    source: Optional[str] = None  # 'crm' | 'outlook' | 'planner' | 'import'

class Appointment(AppointmentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client: Optional[Client] = Relationship(back_populates="appointments")

class AppointmentCreate(AppointmentBase): pass
class AppointmentRead(AppointmentBase):
    id: int

class VisitReportBase(SQLModel):
    appointment_id: int = Field(foreign_key="appointment.id")
    summary: Optional[str] = None
    outcome: Optional[str] = None

class VisitReport(VisitReportBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class VisitReportCreate(VisitReportBase): pass
class VisitReportRead(VisitReportBase):
    id: int

class FollowUpBase(SQLModel):
    client_id: int = Field(foreign_key="client.id")
    title: str
    due_date: datetime
    done: bool = False
    notes: Optional[str] = None

class FollowUp(FollowUpBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    client: Optional[Client] = Relationship(back_populates="followups")

class FollowUpCreate(FollowUpBase): pass
class FollowUpRead(FollowUpBase):
    id: int

# ------------------------------
# Utils
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def nearest_neighbor_route(points: List[Dict], start_index: int = 0) -> List[int]:
    n = len(points)
    unvisited = set(range(n))
    route = [start_index]
    unvisited.remove(start_index)
    current = start_index
    while unvisited:
        nxt = min(unvisited, key=lambda j: haversine(points[current]['lat'], points[current]['lon'], points[j]['lat'], points[j]['lon']))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route

def total_distance(points: List[Dict], order: List[int]) -> float:
    d = 0.0
    for i in range(len(order)-1):
        a, b = points[order[i]], points[order[i+1]]
        d += haversine(a['lat'], a['lon'], b['lat'], b['lon'])
    return d

def two_opt(points: List[Dict], order: List[int]) -> List[int]:
    improved = True
    best = order[:]
    best_dist = total_distance(points, best)
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for k in range(i+1, len(best)-1):
                new_order = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_dist = total_distance(points, new_order)
                if new_dist < best_dist - 1e-6:
                    best, best_dist, improved = new_order, new_dist, True
    return best

def optimize(points: List[Dict], start_index: int = 0) -> List[int]:
    nn = nearest_neighbor_route(points, start_index)
    return two_opt(points, nn)

# ------------------------------
# Startup
# ------------------------------
@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.get("/")
def root():
    return {"ok": True, "name": "CRM Tournées — MVP"}

# ------------------------------
# Clients
# ------------------------------
@app.post("/clients", response_model=ClientRead, tags=["Clients"])
def create_client(data: ClientCreate, session: Session = Depends(get_session)):
    c = Client.model_validate(data)
    session.add(c); session.commit(); session.refresh(c)
    return c

@app.get("/clients", response_model=List[ClientRead], tags=["Clients"])
def list_clients(session: Session = Depends(get_session)):
    return session.exec(select(Client)).all()

@app.get("/clients/{client_id}", response_model=ClientRead, tags=["Clients"])
def get_client(client_id: int, session: Session = Depends(get_session)):
    c = session.get(Client, client_id)
    if not c: raise HTTPException(404, "Client introuvable")
    return c

@app.put("/clients/{client_id}", response_model=ClientRead, tags=["Clients"])
def update_client(client_id: int, data: ClientCreate, session: Session = Depends(get_session)):
    c = session.get(Client, client_id)
    if not c: raise HTTPException(404, "Client introuvable")
    for k, v in data.model_dump().items(): setattr(c, k, v)
    session.add(c); session.commit(); session.refresh(c)
    return c

@app.delete("/clients/{client_id}", tags=["Clients"])
def delete_client(client_id: int, session: Session = Depends(get_session)):
    c = session.get(Client, client_id)
    if not c: raise HTTPException(404, "Client introuvable")
    session.delete(c); session.commit()
    return {"ok": True}

# ------------------------------
# Appointments
# ------------------------------
@app.post("/appointments", response_model=AppointmentRead, tags=["Rendez-vous"])
def create_appointment(data: AppointmentCreate, session: Session = Depends(get_session)):
    a = Appointment.model_validate(data)
    session.add(a); session.commit(); session.refresh(a)
    return a

@app.get("/appointments", response_model=List[AppointmentRead], tags=["Rendez-vous"])
def list_appointments(session: Session = Depends(get_session)):
    return session.exec(select(Appointment)).all()

@app.get("/appointments/{appt_id}", response_model=AppointmentRead, tags=["Rendez-vous"])
def get_appointment(appt_id: int, session: Session = Depends(get_session)):
    a = session.get(Appointment, appt_id)
    if not a: raise HTTPException(404, "Rendez-vous introuvable")
    return a

@app.delete("/appointments/{appt_id}", tags=["Rendez-vous"])
def delete_appointment(appt_id: int, session: Session = Depends(get_session)):
    a = session.get(Appointment, appt_id)
    if not a: raise HTTPException(404, "Rendez-vous introuvable")
    session.delete(a); session.commit()
    return {"ok": True}

# ------------------------------
# Reports
# ------------------------------
@app.post("/reports", response_model=VisitReportRead, tags=["Rapports"])
def create_report(data: VisitReportCreate, session: Session = Depends(get_session)):
    r = VisitReport.model_validate(data)
    session.add(r); session.commit(); session.refresh(r)
    return r

@app.get("/reports", response_model=List[VisitReportRead], tags=["Rapports"])
def list_reports(session: Session = Depends(get_session)):
    return session.exec(select(VisitReport)).all()

# ------------------------------
# Follow-ups
# ------------------------------
@app.post("/followups", response_model=FollowUpRead, tags=["Relances"])
def create_followup(data: FollowUpCreate, session: Session = Depends(get_session)):
    f = FollowUp.model_validate(data)
    session.add(f); session.commit(); session.refresh(f)
    return f

@app.get("/followups", response_model=List[FollowUpRead], tags=["Relances"])
def list_followups(session: Session = Depends(get_session)):
    return session.exec(select(FollowUp)).all()

# ------------------------------
# Planning hebdo (ancre + optimisation)
# ------------------------------
@app.post("/planning/weekly", response_model=List[AppointmentRead], tags=["Tournées"])
def plan_weekly(
    anchor_appt_id: int = Query(..., description="ID d'un rendez-vous existant (ancre)"),
    week_start: datetime = Query(..., description="Début de semaine (YYYY-MM-DD)"),
    daily_start_hour: int = 9,
    meeting_minutes: int = 45,
    travel_buffer_minutes: int = 15,
    max_meetings_per_day: int = 6,
    session: Session = Depends(get_session)
):
    anchor = session.get(Appointment, anchor_appt_id)
    if not anchor: raise HTTPException(404, "Rendez-vous ancre introuvable")
    anchor_client = session.get(Client, anchor.client_id)
    if not anchor_client or (anchor_client.lat is None or anchor_client.lon is None):
        raise HTTPException(400, "Le client ancre doit avoir des coordonnées lat/lon")

    week_end = week_start + timedelta(days=7)
    clients = session.exec(select(Client)).all()
    candidates = []
    for c in clients:
        if c.id == anchor_client.id: 
            continue
        if c.lat is None or c.lon is None:
            continue
        if c.visit_frequency_days:
            last_visit = session.exec(
                select(Appointment).where(Appointment.client_id==c.id, Appointment.start < week_start)
                .order_by(Appointment.start.desc())
            ).first()
            due = True
            if last_visit and c.visit_frequency_days:
                due = (week_start - last_visit.start).days >= c.visit_frequency_days
            if not due:
                continue
        candidates.append(c)

    if not candidates:
        return []

    points = [{"id": c.id, "lat": c.lat, "lon": c.lon} for c in [anchor_client] + candidates]
    order_idx = optimize(points, start_index=0)
    ordered_clients = [next(c for c in [anchor_client] + candidates if c.id == points[i]["id"]) for i in order_idx]

    appts: List[Appointment] = []
    current_day = week_start
    day_count = 0
    per_day = 0
    cur_time = current_day.replace(hour=daily_start_hour, minute=0, second=0, microsecond=0)

    for c in ordered_clients:
        if c.id == anchor_client.id:
            continue
        if per_day >= max_meetings_per_day:
            day_count += 1
            if day_count >= 5:
                break
            current_day = week_start + timedelta(days=day_count)
            per_day = 0
            cur_time = current_day.replace(hour=daily_start_hour, minute=0, second=0, microsecond=0)
        start = cur_time
        end = start + timedelta(minutes=meeting_minutes)
        cur_time = end + timedelta(minutes=travel_buffer_minutes)
        appt = Appointment(
            client_id=c.id, start=start, end=end,
            subject=f"Visite {c.name}",
            location=c.address or c.city or "",
            is_planned=True, source="planner"
        )
        session.add(appt); session.commit(); session.refresh(appt)
        appts.append(appt); per_day += 1
    return appts

# ------------------------------
# ICS / Outlook
# ------------------------------
@app.get("/ics/schedule.ics", response_class=Response, tags=["ICS / Outlook"])
def ics_schedule(session: Session = Depends(get_session)):
    cal = Calendar()
    appts = session.exec(select(Appointment)).all()
    for a in appts:
        ev = Event()
        ev.name = a.subject or f"Visite client #{a.client_id}"
        ev.begin = a.start
        ev.end = a.end
        client = session.get(Client, a.client_id)
        if client:
            ev.location = client.address or client.city or ""
            ev.description = (a.description or "") + (f"\nClient: {client.name}" if client.name else "")
        cal.events.add(ev)
    return Response(str(cal), media_type="text/calendar")
