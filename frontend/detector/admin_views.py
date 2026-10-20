from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group, Permission
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from .forms import AdminUserCreateForm


def _admin_only(request):
    if request.user.is_superuser or request.user.has_perm("detector.can_manage_users") or request.user.has_perm("detector.can_manage_settings"):
        return True
    messages.error(request, "Votre profil ne dispose d'aucun droit d'administration.")
    return False


def _functional_permissions():
    return list(Permission.objects.filter(content_type__app_label="detector", codename__startswith="can_").order_by("codename"))


def _ensure_default_admin_permissions():
    admin_role, _ = Group.objects.get_or_create(name="Administrateur")
    permissions = _functional_permissions()
    if permissions and set(admin_role.permissions.values_list("id", flat=True)) != {p.id for p in permissions}:
        admin_role.permissions.set(permissions)
    return admin_role


def _admin_context():
    User = get_user_model()
    _ensure_default_admin_permissions()
    return {"admin_users_count": User.objects.count(), "admin_roles_count": Group.objects.count(), "admin_active_count": User.objects.filter(is_active=True).count()}


@login_required
def administration(request):
    if not _admin_only(request): return redirect("detector:dashboard")
    return render(request, "administration.html", _admin_context())


@login_required
def administration_roles(request):
    if not (request.user.is_superuser or request.user.has_perm("detector.can_manage_settings")):
        messages.error(request, "Vous ne disposez pas du droit de gestion des rôles et paramètres."); return redirect("detector:dashboard")
    if request.method == "POST":
        name = request.POST.get("role_name", "").strip()
        if not name: messages.error(request, "Le nom du rôle est obligatoire.")
        elif Group.objects.filter(name__iexact=name).exists(): messages.error(request, "Ce rôle existe déjà.")
        else: Group.objects.create(name=name); messages.success(request, f"Le rôle « {name} » a été créé.")
        return redirect("detector:administration_roles")
    roles=Group.objects.prefetch_related("user_set","permissions").order_by("name")
    return render(request,"administration_roles.html",{**_admin_context(),"roles":roles})


@login_required
def administration_role_delete(request,pk):
    if not (request.user.is_superuser or request.user.has_perm("detector.can_manage_settings")):
        messages.error(request,"Vous ne disposez pas du droit de gestion des rôles."); return redirect("detector:dashboard")
    role=get_object_or_404(Group,pk=pk)
    if role.name in ("Utilisateur","Administrateur"): messages.error(request,"Les rôles système Utilisateur et Administrateur ne peuvent pas être supprimés.")
    elif request.method=="POST": role_name=role.name; role.delete(); messages.success(request,f"Le rôle « {role_name} » a été supprimé.")
    return redirect("detector:administration_roles")


@login_required
def administration_users(request):
    if not (request.user.is_superuser or request.user.has_perm("detector.can_manage_users")):
        messages.error(request,"Vous ne disposez pas du droit de gestion des utilisateurs."); return redirect("detector:dashboard")
    User=get_user_model(); q=request.GET.get("q","").strip(); users=User.objects.prefetch_related("groups","user_permissions").order_by("-is_active","username")
    if q: users=users.filter(Q(username__icontains=q)|Q(first_name__icontains=q)|Q(last_name__icontains=q)|Q(email__icontains=q))
    if request.method=="POST":
        form=AdminUserCreateForm(request.POST)
        if form.is_valid():
            role_id=request.POST.get("group_id","").strip()
            role=get_object_or_404(Group,pk=role_id) if role_id else None
            user=form.save(); user.groups.set([role] if role else []); user.is_staff=bool(role and role.name=="Administrateur"); user.save(update_fields=["is_staff"])
            messages.success(request,f"Le compte « {user.username} » a été créé avec le rôle « {role.name if role else 'Aucun rôle'} »."); return redirect("detector:administration_users")
    else: form=AdminUserCreateForm()
    return render(request,"administration_users.html",{**_admin_context(),"users":users,"q":q,"create_user_form":form,"roles":Group.objects.order_by("name")})


@login_required
def administration_permissions(request):
    if not (request.user.is_superuser or request.user.has_perm("detector.can_manage_settings")):
        messages.error(request,"Vous ne disposez pas du droit de gestion des permissions."); return redirect("detector:dashboard")
    admin_role=_ensure_default_admin_permissions(); roles=list(Group.objects.prefetch_related("permissions").order_by("name")); permissions=_functional_permissions()
    if request.method=="POST":
        role=get_object_or_404(Group,pk=request.POST.get("group_id")); selected=set(request.POST.getlist("permissions"))
        if role.name=="Administrateur": _ensure_default_admin_permissions(); messages.info(request,"Le rôle Administrateur conserve automatiquement toutes les permissions fonctionnelles.")
        else: role.permissions.set([p for p in permissions if p.codename in selected]); messages.success(request,f"Les permissions du rôle « {role.name} » ont été mises à jour.")
        return redirect("detector:administration_permissions")
    return render(request,"administration_permissions_global.html",{**_admin_context(),"roles":roles,"permissions":permissions,"administrator_role":admin_role})
