from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group, Permission
from django.shortcuts import get_object_or_404, redirect, render


@login_required
def administration_permissions(request, pk):
    """Attribue à un utilisateur un rôle et des permissions fonctionnelles directes."""
    if not request.user.is_staff:
        messages.error(request, "Accès réservé aux administrateurs.")
        return redirect("detector:dashboard")

    User = get_user_model()
    target = get_object_or_404(User, pk=pk)
    groups = list(Group.objects.prefetch_related("permissions").order_by("name"))
    permissions = list(
        Permission.objects.filter(
            content_type__app_label="detector",
            codename__startswith="can_",
        ).order_by("codename")
    )

    if request.method == "POST":
        group_id = request.POST.get("group_id", "").strip()
        role = get_object_or_404(Group, pk=group_id) if group_id else None
        selected_codes = set(request.POST.getlist("permissions"))
        selected_permissions = [p for p in permissions if p.codename in selected_codes]

        if target.pk == request.user.pk and (not role or role.name != "Administrateur"):
            messages.error(request, "Votre propre compte doit conserver le rôle Administrateur.")
        else:
            if role:
                target.groups.set([role])
            else:
                target.groups.clear()

            target.user_permissions.set(selected_permissions)
            target.is_staff = bool(role and role.name == "Administrateur")
            target.save(update_fields=["is_staff"])

            messages.success(
                request,
                f"Les accès de « {target.username} » ont été enregistrés : "
                f"rôle « {role.name if role else 'Aucun rôle'} » et "
                f"{len(selected_permissions)} permission(s) directe(s).",
            )
            return redirect("detector:administration_permissions", pk=target.pk)

    current_group = target.groups.first()
    direct_permission_ids = set(target.user_permissions.values_list("id", flat=True))
    inherited_permission_ids = set()
    if current_group:
        inherited_permission_ids = set(current_group.permissions.values_list("id", flat=True))

    return render(request, "administration_permissions.html", {
        "target_user": target,
        "groups": groups,
        "permissions": permissions,
        "current_group": current_group,
        "direct_permission_ids": direct_permission_ids,
        "inherited_permission_ids": inherited_permission_ids,
    })
