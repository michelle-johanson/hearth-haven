using HearthHaven.API.Controllers;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Identity;
using HearthHaven.API.Data; // Ensure this matches your exact namespace

var builder = WebApplication.CreateBuilder(args);

// 1. Grab the connection string
var connectionString = builder.Configuration.GetConnectionString("AZURE_SQL_CONNECTIONSTRING");

if (string.IsNullOrWhiteSpace(connectionString))
{
    throw new InvalidOperationException(
        "Missing connection string 'AZURE_SQL_CONNECTIONSTRING'. Configure it with user secrets, environment variables, or hosted application settings.");
}

// 2. Setup your Operational Database
builder.Services.AddDbContext<HearthHavenDbContext>(options =>
    options.UseSqlServer(connectionString));

// 3. Setup your Security Database
builder.Services.AddDbContext<SecurityDbContext>(options =>
    options.UseSqlServer(connectionString));

// 4. Turn on ASP.NET Core Identity
builder.Services.AddIdentity<IdentityUser, IdentityRole>(options => 
{
    options.Password.RequireDigit = true;
    options.Password.RequireLowercase = true;
    options.Password.RequiredLength = 14; 
    options.Password.RequireUppercase = true;
    options.Password.RequireNonAlphanumeric = true;
    options.Password.RequiredUniqueChars = 1;
    options.Lockout.AllowedForNewUsers = true;
    options.Lockout.DefaultLockoutTimeSpan = TimeSpan.FromMinutes(15);
    options.Lockout.MaxFailedAccessAttempts = 5;
})
.AddEntityFrameworkStores<SecurityDbContext>()
.AddDefaultTokenProviders();

// 5. ADDED: Configure CORS policy
var allowedOrigins = builder.Configuration.GetSection("AllowedOrigins").Get<string[]>() ?? [];
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowReactApp", policy =>
    {
        policy.WithOrigins(allowedOrigins)
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials(); // Required if you pass authentication cookies/tokens
    });
});

builder.Services.AddControllers();
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// 6. ADDED: Apply the CORS policy BEFORE Authentication and Authorization
app.UseCors("AllowReactApp");

app.UseAuthentication(); 
app.UseAuthorization();

app.MapControllers();

app.Run();