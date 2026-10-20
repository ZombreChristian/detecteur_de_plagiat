import io
from io import BytesIO
import re
import os
import time
import requests
from pathlib import Path
from django.core.files.storage import FileSystemStorage
from docx import Document as DocxDocument
from docx.shared import RGBColor
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model, update_session_auth_hash
from django.contrib.auth.models import Group, Permission
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.urls import reverse
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.db.models import Q
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from openpyxl import Workbook
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from .forms import (LoginForm, SignUpForm, AdminUserCreateForm, AdminUserUpdateForm, PasswordRecoveryForm, UserPasswordChangeForm)
from .models import Analysis, StudyDocument


def login_view(request):
    if request.user.is_authenticated: return redirect("detector:dashboard")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid(): login(request, form.get_user()); return redirect("detector:dashboard")
    return render(request, "login.html", {"form": form})

def password_reset_request(request):
    if request.user.is_authenticated: return redirect("detector:dashboard")
    form = PasswordRecoveryForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user=form.user; uid=urlsafe_base64_encode(force_bytes(user.pk)); token=default_token_generator.make_token(user); reset_path=reverse("detector:password_reset_confirm",kwargs={"uidb64":uid,"token":token}); base_url=getattr(settings,"PUBLIC_APP_URL","").rstrip("/"); reset_url=f"{base_url}{reset_path}" if base_url else f"{'https' if request.is_secure() else 'http'}://{request.get_host()}{reset_path}"; context={"email":user.email,"user":user,"domain":request.get_host(),"site_name":"TDRDOC-SCAN","protocol":"https" if request.is_secure() else "http","uid":uid,"token":token,"reset_url":reset_url}; subject=render_to_string("password_reset_subject.txt",context).strip(); message=render_to_string("password_reset_email.txt",context); required_email_settings=(getattr(settings,"EMAIL_HOST",""),getattr(settings,"EMAIL_HOST_USER",""),getattr(settings,"EMAIL_HOST_PASSWORD",""),getattr(settings,"DEFAULT_FROM_EMAIL",""))
        if not all(required_email_settings): form.add_error(None,"L'envoi e-mail n'est pas configuré. Renseignez EMAIL_HOST, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD et DEFAULT_FROM_EMAIL dans le fichier .env.")
        else:
            try: send_mail(subject,message,settings.DEFAULT_FROM_EMAIL,[user.email],fail_silently=False)
            except Exception as exc:
                if settings.DEBUG: form.add_error(None,f"Échec de l'envoi SMTP : {exc}")
                else: form.add_error(None,"Le message de récupération n'a pas pu être envoyé. Vérifiez la configuration e-mail du serveur.")
            else: return redirect("detector:password_reset_done")
    return render(request,"password_reset.html",{"form":form})

def password_reset_done(request): return render(request,"password_reset_done.html")

def password_reset_confirm(request,uidb64,token):
    from django.contrib.auth.views import PasswordResetConfirmView
    return PasswordResetConfirmView.as_view(template_name="password_reset_confirm.html",success_url="/mot-de-passe-reinitialise/")(request,uidb64=uidb64,token=token)

def password_reset_complete(request): return render(request,"password_reset_complete.html")

def signup_view(request):
    if request.user.is_authenticated: return redirect("detector:dashboard")
    form=SignUpForm(request.POST or None)
    if request.method=="POST" and form.is_valid(): form.save(); messages.success(request,"Votre compte a été créé. Vous pouvez maintenant vous connecter."); return redirect("detector:login")
    return render(request,"signup.html",{"form":form})

@login_required
def account_settings(request):
    form=UserPasswordChangeForm(request.user,request.POST or None)
    if request.method=="POST" and form.is_valid(): form.save(); update_session_auth_hash(request,request.user); messages.success(request,"Votre mot de passe a été modifié avec succès."); return redirect("detector:account_settings")
    return render(request,"account_settings.html",{"form":form})

def logout_view(request): logout(request); return redirect("detector:login")

def _functional_permissions(): return list(Permission.objects.filter(content_type__app_label="detector",codename__startswith="can_").order_by("codename"))

def _ensure_administrator_role(user=None):
    role,_=Group.objects.get_or_create(name="Administrateur")
    permissions=_functional_permissions()
    if permissions and set(role.permissions.values_list("id",flat=True)) != {p.id for p in permissions}: role.permissions.set(permissions)
    return role

def _has_permission(user,codename):
    if not user or not user.is_authenticated: return False
    if user.is_superuser: return True
    return user.has_perm(f"detector.{codename}")

def _require_permission(request,codename,message):
    if _has_permission(request.user,codename): return True
    messages.error(request,message); return False

@login_required
def administration_utilisateur_supprimer(request,pk):
    if not _require_permission(request,"can_manage_users","Vous ne disposez pas des droits de gestion des utilisateurs."): return redirect("detector:dashboard")
    User=get_user_model(); target=get_object_or_404(User,pk=pk)
    if target.pk==request.user.pk: messages.error(request,"Vous ne pouvez pas supprimer votre propre compte."); return redirect("detector:administration_users")
    if request.method=="POST": target.is_active=False; target.save(update_fields=["is_active"]); messages.success(request,f"Le compte « {target.username} » a été désactivé."); return redirect("detector:administration_users")
    return render(request,"administration_utilisateur.html",{"target_user":target,"profile_form":AdminUserUpdateForm(instance=target),"groups":Group.objects.order_by("name"),"current_group":target.groups.first(),"delete_confirmation":True})

def _get_uploaded_docx(request):
    for field_name in ("document","tdr_document","report_document","file"):
        uploaded=request.FILES.get(field_name)
        if uploaded and uploaded.name: return uploaded
    for uploaded in request.FILES.values():
        if uploaded and uploaded.name: return uploaded
    return None

def _validate_document(uploaded): return bool(uploaded and uploaded.name and Path(uploaded.name).suffix.lower() in {".docx",".pdf"})
def _document_content_type(uploaded): return "application/pdf" if Path(uploaded.name).suffix.lower()==".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

def _save_uploaded_document(uploaded):
    media_root=Path(settings.MEDIA_ROOT)/"documents"; media_root.mkdir(parents=True,exist_ok=True); storage=FileSystemStorage(location=str(media_root)); uploaded.seek(0); saved_name=storage.save(uploaded.name,uploaded); uploaded.seek(0); return f"documents/{saved_name}"
def _document_url(relative_path): return f"{settings.MEDIA_URL.rstrip('/')}/{relative_path.lstrip('/')}" if relative_path else ""
def _safe_reference_path(document):
    if not document or not document.file_path: return None
    root=Path(settings.BASE_DIR).resolve().parent; candidate=(root/document.file_path).resolve()
    try: candidate.relative_to(root)
    except ValueError: return None
    return candidate if candidate.is_file() else None


def _read_saved_document(path):
    if not path: return ""
    file_path=Path(settings.MEDIA_ROOT)/str(path).replace("/",os.sep)
    if not file_path.exists() or file_path.suffix.lower()!=".docx": return ""
    doc=DocxDocument(file_path); parts=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells=[cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells: parts.append(" | ".join(cells))
    return "\n\n".join(parts)

def _read_document_from_corpus(filename,mode):
    root=Path(settings.BASE_DIR).resolve().parent; folder=root/"donnees"/("TDR" if mode=="duplicate" else "Rapport d'etude")
    if not folder.exists(): return ""
    target=Path(filename).name.strip().lower()
    for path in folder.rglob("*.docx"):
        if path.name.strip().lower()==target:
            doc=DocxDocument(path); parts=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells=[cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells: parts.append(" | ".join(cells))
            return "\n\n".join(parts)
    return ""

def _analysis_problem_passages(result):
    passages=[]; seen=set()
    for group in result.get("matches",[]) or []:
        for match in group.get("matches",[]) or []:
            passage=(match.get("candidate_passage") or "").strip()
            if passage and passage not in seen: seen.add(passage); passages.append(passage)
    return sorted(passages,key=len,reverse=True)

def _normalized_with_map(text):
    normalized=[]; positions=[]; previous_space=False
    for index,char in enumerate(text or ""):
        if char.isspace():
            if not previous_space: normalized.append(" "); positions.append(index)
            previous_space=True; continue
        normalized.append(char.casefold()); positions.append(index); previous_space=False
    while normalized and normalized[0]==" ": normalized.pop(0); positions.pop(0)
    while normalized and normalized[-1]==" ": normalized.pop(); positions.pop()
    return "".join(normalized),positions

def _highlight_text_runs(paragraph,passages):
    text=paragraph.text
    if not text or not passages: return
    normalized_text,positions=_normalized_with_map(text); fragments=[]
    for passage in passages:
        normalized_passage,_=_normalized_with_map(passage)
        if not normalized_passage: continue
        candidates=[normalized_passage]
        for fragment in re.split(r"[.!?;:]+",passage):
            fragment_norm,_=_normalized_with_map(fragment)
            if len(fragment_norm)>=45: candidates.append(fragment_norm)
        if len(normalized_passage)>140:
            for start in range(0,len(normalized_passage)-59,60):
                chunk=normalized_passage[start:start+100].strip()
                if len(chunk)>=45: candidates.append(chunk)
        for candidate in candidates:
            start=0
            while candidate:
                index=normalized_text.find(candidate,start)
                if index<0: break
                end=index+len(candidate)-1
                if index<len(positions) and end<len(positions): fragments.append((positions[index],positions[end]+1))
                start=index+max(1,len(candidate))
    if not fragments: return
    fragments.sort(); merged=[]
    for start,end in fragments:
        if not merged or start>merged[-1][1]: merged.append([start,end])
        else: merged[-1][1]=max(merged[-1][1],end)
    for run in list(paragraph.runs): run._element.getparent().remove(run._element)
    cursor=0
    for start,end in merged:
        if start>cursor: paragraph.add_run(text[cursor:start])
        run=paragraph.add_run(text[start:end]); run.font.color.rgb=RGBColor(0xC0,0x00,0x00); run.bold=True; cursor=end
    if cursor<len(text): paragraph.add_run(text[cursor:])

def _highlight_docx(path,passages):
    document=DocxDocument(path)
    for paragraph in document.paragraphs: _highlight_text_runs(paragraph,passages)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs: _highlight_text_runs(paragraph,passages)
    output=BytesIO(); document.save(output); output.seek(0); return output

def _pdf_text(path):
    reader=PdfReader(str(path)); return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages if (page.extract_text() or "").strip())
def _highlight_pdf(path,passages,title):
    from xml.sax.saxutils import escape
    output=BytesIO(); doc=SimpleDocTemplate(output,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42,title=title); normal=ParagraphStyle("AnnotatedNormal",fontName="Helvetica",fontSize=9.5,leading=13,alignment=TA_LEFT,spaceAfter=8); title_style=ParagraphStyle("AnnotatedTitle",parent=normal,fontSize=14,leading=18,spaceAfter=16); story=[Paragraph(escape("TDRDOC-SCAN — Document analysé avec passages signalés"),title_style)]; text=_pdf_text(path)
    for block in re.split(r"\n{2,}",text):
        block=block.strip()
        if not block: continue
        cursor=0; matches=[]
        for passage in passages:
            start=0
            while True:
                index=block.find(passage,start)
                if index<0: break
                matches.append((index,index+len(passage))); start=index+len(passage)
        matches.sort(); merged=[]
        for start,end in matches:
            if not merged or start>merged[-1][1]: merged.append([start,end])
            else: merged[-1][1]=max(merged[-1][1],end)
        pieces=[]
        for start,end in merged:
            if start>cursor: pieces.append(escape(block[cursor:start]))
            pieces.append(f'<font color="#C00000"><b>{escape(block[start:end])}</b></font>'); cursor=end
        if cursor<len(block): pieces.append(escape(block[cursor:]))
        story.append(Paragraph("".join(pieces),normal))
    doc.build(story); output.seek(0); return output

@login_required
def download_annotated_document(request,pk):
    if not _require_permission(request,"can_view_history","Votre profil ne dispose pas de l'accès à l'historique."): return redirect("detector:dashboard")
    analysis=get_object_or_404(Analysis,pk=pk,user=request.user); result=analysis.result_json or {}; uploaded_path=result.get("uploaded_path","")
    if not uploaded_path: messages.error(request,"Le document original de cette analyse n'est plus disponible."); return redirect("detector:analysis_detail",pk=analysis.pk)
    path=(Path(settings.MEDIA_ROOT)/str(uploaded_path).replace("/",os.sep)).resolve(); media_root=Path(settings.MEDIA_ROOT).resolve()
    try: path.relative_to(media_root)
    except ValueError: messages.error(request,"Le fichier demandé n'est pas autorisé."); return redirect("detector:analysis_detail",pk=analysis.pk)
    if not path.is_file(): messages.error(request,"Le document original de cette analyse n'est plus disponible."); return redirect("detector:analysis_detail",pk=analysis.pk)
    passages=_analysis_problem_passages(result); stem=path.stem
    if path.suffix.lower()==".docx": return FileResponse(_highlight_docx(path,passages),as_attachment=True,filename=f"{stem}_passages_signales.docx",content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    if path.suffix.lower()==".pdf": return FileResponse(_highlight_pdf(path,passages,f"{stem}_passages_signales"),as_attachment=True,filename=f"{stem}_passages_signales.pdf",content_type="application/pdf")
    messages.error(request,"Format de document non pris en charge."); return redirect("detector:analysis_detail",pk=analysis.pk)

@login_required
def analysis_detail(request,pk):
    if not _require_permission(request,"can_view_history","Votre profil ne dispose pas de l'accès à l'historique."): return redirect("detector:dashboard")
    analysis=get_object_or_404(Analysis,pk=pk,user=request.user); result=analysis.result_json or {}; uploaded_path=result.get("uploaded_path",""); return render(request,"analysis_detail.html",{"analysis":analysis,"result":result,"document_url":_document_url(uploaded_path),"document_path":uploaded_path})

@login_required
def reception(request):
    if not _require_permission(request,"can_verify_tdr","Votre profil ne dispose pas de l'accès à la vérification des TDR."): return redirect("detector:dashboard")
    if request.method=="GET": return render(request,"reception.html")
    mode=request.POST.get("mode","tdr"); uploaded=_get_uploaded_docx(request)
    if not _validate_document(uploaded): messages.error(request,"Veuillez sélectionner un document Word (.docx) ou PDF (.pdf) valide."); return redirect("detector:reception")
    uploaded_path=_save_uploaded_document(uploaded)
    try: threshold=float(request.POST.get("threshold",""))/100.0; assert 0.01<=threshold<=1.0
    except (TypeError,ValueError,AssertionError): messages.error(request,"Le seuil doit être compris entre 1 % et 100 %."); return redirect("detector:reception")
    endpoint="/api/detect/duplicate" if mode=="tdr" else "/api/detect/plagiarism"; start=time.perf_counter()
    try:
        response=requests.post(f"{settings.FASTAPI_URL}{endpoint}",data={"threshold":threshold},files={"file":(uploaded.name,uploaded.file,uploaded.content_type or _document_content_type(uploaded))},timeout=600); response.raise_for_status(); result=response.json(); result["uploaded_path"]=uploaded_path; duration=round((time.perf_counter()-start)*1000); result["duration_ms"]=duration
        analysis=Analysis.objects.create(user=request.user,document_name=uploaded.name,mode="duplicate" if mode=="tdr" else "plagiarism",decision=result.get("decision",""),hybrid_score=result.get("hybrid_score"),semantic_score=result.get("semantic_score"),tfidf_score=result.get("tfidf_score"),novelty_score=result.get("novelty_score"),result_json=result,duration_ms=duration)
        request.session["last_result"]=result; request.session["last_analysis_id"]=analysis.id; request.session["last_document_name"]=uploaded.name
        if mode=="tdr" and result.get("decision")=="DIFFERENT": request.session["validated_tdr_name"]=uploaded.name; messages.success(request,"TDR accepté : aucune similarité suffisante n'a été trouvée. Le document n'a pas été ajouté automatiquement au registre."); return redirect("detector:reception")
        if mode=="tdr" and result.get("decision")=="SIMILAIRE": messages.error(request,"TDR non retenu : un document similaire existe déjà dans la base.")
        else: messages.info(request,"Rapport analysé. Consultez le détail des résultats.")
        return redirect("detector:resultats")
    except requests.RequestException as exc: messages.error(request,f"Le moteur d'analyse est indisponible : {exc}")
    return redirect("detector:reception")

@login_required
def upload_report_for_tdr(request,pk):
    if not _require_permission(request,"can_analyze_report","Votre profil ne dispose pas de l'accès à l'analyse des rapports."): return redirect("detector:dashboard")
    tdr=get_object_or_404(Analysis,pk=pk,user=request.user,mode="duplicate",decision="DIFFERENT")
    if request.method=="GET": return render(request,"report_upload.html",{"tdr":tdr})
    uploaded=_get_uploaded_docx(request)
    if not _validate_document(uploaded): messages.error(request,"Veuillez sélectionner un rapport Word (.docx) ou PDF (.pdf) valide."); return redirect("detector:upload_report_for_tdr",pk=tdr.pk)
    uploaded_path=_save_uploaded_document(uploaded)
    try: threshold=float(request.POST.get("threshold",""))/100.0; assert 0.01<=threshold<=1.0
    except (TypeError,ValueError,AssertionError): messages.error(request,"Le seuil doit être compris entre 1 % et 100 %."); return redirect("detector:upload_report_for_tdr",pk=tdr.pk)
    start=time.perf_counter()
    try:
        response=requests.post(f"{settings.FASTAPI_URL}/api/detect/plagiarism",data={"threshold":threshold},files={"file":(uploaded.name,uploaded.file,uploaded.content_type or _document_content_type(uploaded))},timeout=600); response.raise_for_status(); result=response.json(); result["uploaded_path"]=uploaded_path; result["linked_tdr_analysis_id"]=tdr.id; result["linked_tdr_name"]=tdr.document_name; duration=round((time.perf_counter()-start)*1000); result["duration_ms"]=duration
        report=Analysis.objects.create(user=request.user,document_name=uploaded.name,mode="plagiarism",decision=result.get("decision",""),hybrid_score=result.get("hybrid_score"),semantic_score=result.get("semantic_score"),tfidf_score=result.get("tfidf_score"),novelty_score=result.get("novelty_score"),result_json=result,duration_ms=duration); request.session["last_result"]=result; request.session["last_analysis_id"]=report.id; request.session["last_document_name"]=uploaded.name; messages.success(request,"Rapport associé au TDR et analysé avec succès."); return redirect("detector:resultats")
    except requests.RequestException as exc: messages.error(request,f"Le moteur FastAPI est indisponible : {exc}"); return redirect("detector:upload_report_for_tdr",pk=tdr.pk)

@login_required
def analyser(request):
    if not _require_permission(request,"can_analyze_report","Votre profil ne dispose pas de l'accès à l'analyse des rapports."): return redirect("detector:dashboard")
    if request.method!="POST": return render(request,"analyser.html")
    uploaded=_get_uploaded_docx(request); mode=request.POST.get("mode","plagiarism")
    if mode not in ("plagiarism","duplicate"): messages.error(request,"Mode d'analyse invalide."); return redirect("detector:analyser")
    if not _validate_document(uploaded): messages.error(request,"Veuillez sélectionner un document Word (.docx) ou PDF (.pdf) valide."); return redirect("detector:analyser")
    uploaded_path=_save_uploaded_document(uploaded)
    try: threshold=float(request.POST.get("threshold",""))/100.0; assert 0.01<=threshold<=1.0
    except (TypeError,ValueError,AssertionError): messages.error(request,"Le seuil doit être compris entre 1 % et 100 %."); return redirect("detector:analyser")
    endpoint="/api/detect/duplicate" if mode=="duplicate" else "/api/detect/plagiarism"; start=time.perf_counter()
    try:
        response=requests.post(f"{settings.FASTAPI_URL}{endpoint}",data={"threshold":threshold},files={"file":(uploaded.name,uploaded.file,uploaded.content_type or _document_content_type(uploaded))},timeout=600); response.raise_for_status(); result=response.json(); result["uploaded_path"]=uploaded_path; duration=round((time.perf_counter()-start)*1000); result["duration_ms"]=duration; analysis=Analysis.objects.create(user=request.user,document_name=uploaded.name,mode=mode,decision=result.get("decision",""),hybrid_score=result.get("hybrid_score"),semantic_score=result.get("semantic_score"),tfidf_score=result.get("tfidf_score"),novelty_score=result.get("novelty_score"),result_json=result,duration_ms=duration); request.session["last_result"]=result; request.session["last_analysis_id"]=analysis.id; request.session["last_document_name"]=uploaded.name; return redirect("detector:resultats")
    except requests.RequestException as exc: messages.error(request,f"Le moteur FastAPI est indisponible : {exc}")
    except Exception as exc: messages.error(request,f"Erreur pendant l'analyse : {exc}")
    return render(request,"analyser.html")

@login_required
def delete_analysis(request,pk):
    if not _require_permission(request,"can_view_history","Votre profil ne dispose pas de l'accès à l'historique."): return redirect("detector:dashboard")
    analysis=get_object_or_404(Analysis,pk=pk,user=request.user)
    if request.method!="POST": return render(request,"analysis_delete_confirm.html",{"analysis":analysis})
    uploaded_path=(analysis.result_json or {}).get("uploaded_path","")
    if uploaded_path:
        path=(Path(settings.MEDIA_ROOT)/str(uploaded_path).replace("/",os.sep)).resolve(); media_root=Path(settings.MEDIA_ROOT).resolve()
        try: path.relative_to(media_root); path.unlink() if path.is_file() else None
        except (ValueError,OSError): pass
    analysis.delete(); messages.success(request,"L'analyse a été supprimée de votre historique."); return redirect("detector:resultats")

@login_required
def resultats(request):
    if not _require_permission(request,"can_view_history","Votre profil ne dispose pas de l'accès à l'historique."): return redirect("detector:dashboard")
    analyses=Analysis.objects.filter(user=request.user).order_by("-created_at"); q=request.GET.get("q","").strip(); decision=request.GET.get("decision","all"); mode=request.GET.get("mode","all")
    if q: analyses=analyses.filter(Q(document_name__icontains=q)|Q(result_json__best_source__icontains=q))
    if decision in ("SIMILAIRE","DIFFERENT","À EXAMINER"): analyses=analyses.filter(decision=decision)
    if mode in ("duplicate","plagiarism"): analyses=analyses.filter(mode=mode)
    paginator=Paginator(analyses,10); page_obj=paginator.get_page(request.GET.get("page",1)); return render(request,"resultats.html",{"analyses":page_obj.object_list,"page_obj":page_obj,"total_filtered":paginator.count,"q":q,"selected_decision":decision,"selected_mode":mode})

@login_required
def export_excel(request):
    if not _require_permission(request,"can_export_results","Votre profil ne dispose pas du droit d'exporter les résultats."): return redirect("detector:dashboard")
    wb=Workbook(); ws=wb.active; ws.title="Analyses"; ws.append(["Document","Mode","Décision","Score hybride","Score sémantique","TF-IDF","Date"])
    for a in Analysis.objects.filter(user=request.user).order_by("-created_at"): ws.append([a.document_name,a.get_mode_display(),a.decision,a.hybrid_score,a.semantic_score,a.tfidf_score,a.created_at.strftime("%d/%m/%Y %H:%M")])
    output=BytesIO(); wb.save(output); output.seek(0); response=HttpResponse(output.getvalue(),content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); response["Content-Disposition"]='attachment; filename="tdrdoc_scan_analyses.xlsx"'; return response

@login_required
def export_pdf(request):
    if not _require_permission(request,"can_export_results","Votre profil ne dispose pas du droit d'exporter les résultats."): return redirect("detector:dashboard")
    output=BytesIO(); doc=SimpleDocTemplate(output,pagesize=A4,rightMargin=32,leftMargin=32,topMargin=32,bottomMargin=32); styles=getSampleStyleSheet(); story=[Paragraph("TDRDOC-SCAN — Rapport d'analyses",styles["Title"]),Spacer(1,12)]; data=[["Document","Mode","Décision","Score hybride","Date"]]
    for a in Analysis.objects.filter(user=request.user).order_by("-created_at"): data.append([a.document_name,a.get_mode_display(),a.decision,f"{(a.hybrid_score or 0):.3f}",a.created_at.strftime("%d/%m/%Y %H:%M")])
    table=Table(data,repeatRows=1); table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#006B3F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP")])); story.append(table); doc.build(story); output.seek(0); response=HttpResponse(output.getvalue(),content_type="application/pdf"); response["Content-Disposition"]='attachment; filename="tdrdoc_scan_analyses.pdf"'; return response

@login_required
def administration_recovery(request):
    User=get_user_model(); active_admin=User.objects.filter(is_superuser=True,is_active=True).exists() or User.objects.filter(groups__name="Administrateur",is_active=True).exists()
    if active_admin: return redirect("detector:dashboard")
    if request.method=="POST":
        username=request.POST.get("username","").strip(); email=request.POST.get("email","").strip().lower(); user=User.objects.filter(username=username,email__iexact=email,is_active=True).first()
        if not user: messages.error(request,"Aucun compte actif correspondant n'a été trouvé.")
        else:
            role=_ensure_administrator_role(); user.groups.add(role); user.is_staff=True; user.save(update_fields=["is_staff"]); messages.success(request,"Votre compte a récupéré les droits Administrateur."); return redirect("detector:dashboard")
    return render(request,"administration_recovery.html")

@login_required
def administration_utilisateur(request,pk):
    if not _require_permission(request,"can_manage_users","Vous ne disposez pas des droits de gestion des utilisateurs."): return redirect("detector:dashboard")
    User=get_user_model(); target=get_object_or_404(User,pk=pk); groups=Group.objects.order_by("name"); current_group=target.groups.first(); form=AdminUserUpdateForm(request.POST or None,instance=target)
    if request.method=="POST" and form.is_valid():
        target=form.save(commit=False); target.is_staff=bool(target.groups.filter(name="Administrateur").exists()); target.save(); role_id=request.POST.get("group_id","").strip(); role=get_object_or_404(Group,pk=role_id) if role_id else None; target.groups.set([role] if role else []); target.is_staff=bool(role and role.name=="Administrateur"); target.save(update_fields=["is_staff"]); messages.success(request,f"Le profil « {target.username} » a été mis à jour avec le rôle « {role.name if role else 'Aucun rôle'} »."); return redirect("detector:administration_users")
    return render(request,"administration_utilisateur.html",{"target_user":target,"profile_form":form,"groups":groups,"current_group":current_group})
